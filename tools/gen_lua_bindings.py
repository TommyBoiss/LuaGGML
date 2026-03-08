#!/usr/bin/env python3
"""Generate Lua C bindings for ggml APIs using clang JSON AST.

This script scans one or more ggml headers and emits:
- lua_ggml_bindings.c: Lua wrappers, record helpers, and constant exports
- lua_ggml_bindings.h: registration function declaration

Only signatures supported by the built-in type map are generated.
Unsupported functions are skipped with a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EXCLUDED_BY_DEFAULT: set[str] = set()

STRUCT_INTEGER_SHORTHANDS = {
    # Preserve the existing user-facing API: ggml.init(mem_size)
    "ggml_init_params": "mem_size",
}

INTEGER_NAMES = {
    "char",
    "signed char",
    "unsigned char",
    "short",
    "short int",
    "signed short",
    "signed short int",
    "unsigned short",
    "unsigned short int",
    "int",
    "signed",
    "signed int",
    "unsigned",
    "unsigned int",
    "long",
    "long int",
    "signed long",
    "signed long int",
    "unsigned long",
    "unsigned long int",
    "long long",
    "long long int",
    "signed long long",
    "signed long long int",
    "unsigned long long",
    "unsigned long long int",
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "intptr_t",
    "uintptr_t",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
}

FLOAT_NAMES = {
    "float",
    "double",
    "long double",
}


@dataclass(frozen=True)
class TypeInfo:
    kind: str
    c_spelling: str
    record_name: Optional[str] = None


@dataclass(frozen=True)
class FieldInfo:
    name: str
    type_info: TypeInfo


@dataclass(frozen=True)
class RecordInfo:
    name: str
    c_spelling: str
    fields: Tuple[FieldInfo, ...]


@dataclass(frozen=True)
class ParamInfo:
    name: str
    type_info: TypeInfo


@dataclass(frozen=True)
class FuncInfo:
    c_name: str
    lua_name: str
    return_type: TypeInfo
    params: Tuple[ParamInfo, ...]


@dataclass(frozen=True)
class ConstantInfo:
    c_name: str
    lua_name: str
    value: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lua bindings for ggml using clang JSON AST")
    parser.add_argument(
        "--header",
        action="append",
        required=True,
        help="Path to a ggml header to scan (repeatable)",
    )
    parser.add_argument("--out-c", required=True, help="Output C file path")
    parser.add_argument("--out-h", required=True, help="Output header file path")
    parser.add_argument(
        "--clang-bin",
        default="clang",
        help="Clang executable used to produce JSON AST",
    )
    parser.add_argument(
        "--clang-arg",
        action="append",
        default=[],
        help="Extra argument passed to clang parser (repeatable)",
    )
    parser.add_argument(
        "--prefix",
        default="ggml_",
        help="Function prefix to include from headers",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Function name to exclude from generation (repeatable)",
    )
    parser.add_argument(
        "--nm-library",
        action="append",
        default=[],
        help="Built library to inspect for exported symbols (repeatable)",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def canonical_text(type_text: str) -> str:
    return " ".join(type_text.replace("\t", " ").split())


def extract_return_type(function_type_text: str) -> str:
    idx = function_type_text.find("(")
    if idx == -1:
        return canonical_text(function_type_text)
    return canonical_text(function_type_text[:idx])


def record_helper_suffix(record_name: str) -> str:
    return record_name.replace(" ", "_")


def constant_alias(constant_name: str) -> str:
    if constant_name.startswith("GGML_"):
        return constant_name[5:]
    return constant_name


def resolve_alias_text(type_text: str, typedefs: Dict[str, str]) -> str:
    resolved = canonical_text(type_text)
    seen: set[str] = set()

    while resolved in typedefs and resolved not in seen:
        seen.add(resolved)
        resolved = canonical_text(typedefs[resolved])

    return resolved


def collect_typedefs(ast_roots: Sequence[dict]) -> Dict[str, str]:
    typedefs: Dict[str, str] = {}

    for ast_root in ast_roots:
        for node in walk_nodes(ast_root):
            if node.get("kind") != "TypedefDecl":
                continue

            name = node.get("name", "")
            underlying = canonical_text(node.get("type", {}).get("qualType", ""))
            if not name or not underlying or name in typedefs:
                continue

            typedefs[name] = underlying

    return typedefs


def collect_exported_symbols(libraries: Sequence[str]) -> set[str]:
    exported: set[str] = set()

    for library in libraries:
        if not os.path.exists(library):
            raise RuntimeError(f"export symbol library not found: {library}")

        proc = subprocess.run(
            ["nm", "-g", "--defined-only", library],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr.strip(), file=sys.stderr)
            raise RuntimeError(f"failed to inspect exported symbols for {library}")

        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                exported.add(parts[-1])

    return exported


def classify_type(qual: str, desugared: str, typedefs: Dict[str, str]) -> Optional[TypeInfo]:
    qual_clean = canonical_text(qual)
    desugared_clean = canonical_text(desugared) if desugared else qual_clean
    qual_resolved = resolve_alias_text(qual_clean, typedefs)
    desugared_resolved = resolve_alias_text(desugared_clean, typedefs)

    if not qual_clean:
        return None

    if "[" in qual_resolved or "[" in desugared_resolved:
        return None

    if qual_clean == "void":
        return TypeInfo("void", qual_clean)

    if desugared_resolved == "_Bool" or qual_clean == "bool":
        return TypeInfo("bool", qual_clean)

    if qual_clean.startswith("enum ") or desugared_resolved.startswith("enum "):
        return TypeInfo("enum", qual_clean)

    if "(*" in desugared_resolved or "(*" in qual_resolved:
        return None

    if "*" in desugared_resolved or "*" in qual_resolved:
        star_count = desugared_resolved.count("*") if "*" in desugared_resolved else qual_resolved.count("*")
        if star_count > 1:
            return None

        no_space = desugared_resolved.replace(" ", "")
        if no_space in {"constchar*", "charconst*"}:
            return TypeInfo("string", qual_clean)

        return TypeInfo("pointer", qual_clean)

    record_text = ""
    if qual_clean.startswith("struct "):
        record_text = qual_clean
    elif desugared_resolved.startswith("struct "):
        record_text = desugared_resolved

    if record_text:
        return TypeInfo("record", qual_clean, record_text.split(" ", 1)[1])

    if desugared_resolved in INTEGER_NAMES or qual_clean in INTEGER_NAMES:
        return TypeInfo("integer", qual_clean)

    if desugared_resolved in FLOAT_NAMES or qual_clean in FLOAT_NAMES:
        return TypeInfo("number", qual_clean)

    return None


def walk_nodes(node: dict) -> Iterable[dict]:
    yield node
    for child in node.get("inner", []):
        yield from walk_nodes(child)


def has_deprecated_attr(node: dict) -> bool:
    return any(child.get("kind") == "DeprecatedAttr" for child in node.get("inner", []))


def get_param_name(param_node: dict, index: int) -> str:
    name = param_node.get("name", "")
    if name:
        return name
    return f"arg{index}"


def analyze_record_candidate(node: dict, typedefs: Dict[str, str]) -> Optional[RecordInfo]:
    name = node.get("name", "")
    if not name:
        return None

    field_nodes = [child for child in node.get("inner", []) if child.get("kind") == "FieldDecl"]
    if not field_nodes:
        return None

    fields: List[FieldInfo] = []
    for field_node in field_nodes:
        field_name = field_node.get("name", "")
        if not field_name:
            return None

        field_type = field_node.get("type", {})
        qual = canonical_text(field_type.get("qualType", ""))
        desugared = canonical_text(field_type.get("desugaredQualType", qual))
        mapped = classify_type(qual, desugared, typedefs)
        if mapped is None:
            return None
        fields.append(FieldInfo(field_name, mapped))

    return RecordInfo(name, f"struct {name}", tuple(fields))


def collect_record_candidates(ast_roots: Sequence[dict], typedefs: Dict[str, str]) -> Dict[str, RecordInfo]:
    candidates: Dict[str, RecordInfo] = {}

    for ast_root in ast_roots:
        for node in walk_nodes(ast_root):
            if node.get("kind") != "RecordDecl":
                continue

            candidate = analyze_record_candidate(node, typedefs)
            if candidate is None:
                continue

            if candidate.name not in candidates:
                candidates[candidate.name] = candidate

    return candidates


def resolve_records(candidates: Dict[str, RecordInfo]) -> Dict[str, RecordInfo]:
    supported: Dict[str, RecordInfo] = {}
    progress = True

    while progress:
        progress = False
        for name, record in candidates.items():
            if name in supported:
                continue

            unresolved = False
            for field in record.fields:
                if field.type_info.kind == "record" and field.type_info.record_name not in supported:
                    unresolved = True
                    break

            if unresolved:
                continue

            supported[name] = record
            progress = True

    return supported


def analyze_function(
    fn_node: dict,
    prefix: str,
    excluded: set[str],
    records: Dict[str, RecordInfo],
    typedefs: Dict[str, str],
) -> Tuple[Optional[FuncInfo], Optional[str]]:
    c_name = fn_node.get("name", "")
    if not c_name or not c_name.startswith(prefix):
        return None, None

    if c_name in excluded:
        return None, "excluded"

    if has_deprecated_attr(fn_node):
        return None, "deprecated function"

    fn_type = fn_node.get("type", {})
    fn_qual = canonical_text(fn_type.get("qualType", ""))
    fn_desugared = canonical_text(fn_type.get("desugaredQualType", fn_qual))

    if "..." in fn_qual or "..." in fn_desugared:
        return None, "variadic function"

    ret_qual = extract_return_type(fn_qual)
    ret_desugared = extract_return_type(fn_desugared)
    ret = classify_type(ret_qual, ret_desugared, typedefs)
    if ret is None:
        return None, f"unsupported return type '{ret_qual}'"
    if ret.kind == "record" and ret.record_name not in records:
        return None, f"unsupported return type '{ret_qual}'"

    params: List[ParamInfo] = []
    param_nodes = [child for child in fn_node.get("inner", []) if child.get("kind") == "ParmVarDecl"]
    for i, param_node in enumerate(param_nodes, start=1):
        ptype = param_node.get("type", {})
        p_qual = canonical_text(ptype.get("qualType", ""))
        p_desugared = canonical_text(ptype.get("desugaredQualType", p_qual))
        mapped = classify_type(p_qual, p_desugared, typedefs)
        if mapped is None:
            return None, f"unsupported param type '{p_qual}'"
        if mapped.kind == "record" and mapped.record_name not in records:
            return None, f"unsupported param type '{p_qual}'"
        params.append(ParamInfo(get_param_name(param_node, i), mapped))

    lua_name = c_name[len(prefix) :] or c_name
    return FuncInfo(c_name, lua_name, ret, tuple(params)), None


def find_functions(
    ast_roots: Sequence[dict],
    prefix: str,
    excluded: set[str],
    records: Dict[str, RecordInfo],
    typedefs: Dict[str, str],
    exported_symbols: set[str],
) -> Tuple[List[FuncInfo], Dict[str, int]]:
    seen_supported: Dict[str, FuncInfo] = {}
    seen_skipped: set[str] = set()
    skipped: Dict[str, int] = {}

    for ast_root in ast_roots:
        for node in walk_nodes(ast_root):
            if node.get("kind") != "FunctionDecl":
                continue

            c_name = node.get("name", "")
            if c_name and c_name in seen_supported:
                continue

            info, reason = analyze_function(node, prefix, excluded, records, typedefs)
            if info is None:
                if reason and c_name and c_name not in seen_skipped:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    seen_skipped.add(c_name)
                continue

            if exported_symbols and info.c_name not in exported_symbols:
                if info.c_name not in seen_skipped:
                    skipped["missing exported symbol"] = skipped.get("missing exported symbol", 0) + 1
                    seen_skipped.add(info.c_name)
                continue

            seen_supported[info.c_name] = info

    functions = sorted(seen_supported.values(), key=lambda fn: fn.c_name)

    alias_counts: Dict[str, int] = {}
    for fn in functions:
        alias_counts[fn.lua_name] = alias_counts.get(fn.lua_name, 0) + 1

    normalized: List[FuncInfo] = []
    for fn in functions:
        if alias_counts[fn.lua_name] > 1:
            normalized.append(
                FuncInfo(
                    c_name=fn.c_name,
                    lua_name=fn.c_name,
                    return_type=fn.return_type,
                    params=fn.params,
                )
            )
            skipped["lua name collision"] = skipped.get("lua name collision", 0) + 1
        else:
            normalized.append(fn)

    return normalized, skipped


def parse_constant_value(node: dict) -> Optional[int]:
    if "value" in node:
        try:
            return int(str(node["value"]), 0)
        except ValueError:
            return None

    for child in node.get("inner", []):
        value = parse_constant_value(child)
        if value is not None:
            return value

    return None


def find_constants(ast_roots: Sequence[dict]) -> Tuple[List[ConstantInfo], Dict[str, int]]:
    constants: Dict[str, ConstantInfo] = {}

    for ast_root in ast_roots:
        for node in walk_nodes(ast_root):
            if node.get("kind") != "EnumConstantDecl":
                continue

            c_name = node.get("name", "")
            if not c_name.startswith("GGML_") or c_name in constants:
                continue

            value = parse_constant_value(node)
            if value is None:
                continue

            constants[c_name] = ConstantInfo(c_name, constant_alias(c_name), value)

    ordered = sorted(constants.values(), key=lambda item: item.c_name)
    alias_counts: Dict[str, int] = {}
    for item in ordered:
        alias_counts[item.lua_name] = alias_counts.get(item.lua_name, 0) + 1

    return ordered, alias_counts


def render_pointer_read_expr(c_type: str, index_expr: str, label: str) -> str:
    return f"({c_type}) lua_ggml_to_pointer(L, {index_expr}, \"{label}\")"


def render_param_read_expr(type_info: TypeInfo, name: str, index: int) -> List[str]:
    c_type = type_info.c_spelling

    if type_info.kind in {"integer", "enum"}:
        return [f"    {c_type} {name} = ({c_type}) luaL_checkinteger(L, {index});"]

    if type_info.kind == "number":
        return [f"    {c_type} {name} = ({c_type}) luaL_checknumber(L, {index});"]

    if type_info.kind == "bool":
        return [f"    {c_type} {name} = ({c_type}) (lua_toboolean(L, {index}) != 0);"]

    if type_info.kind == "string":
        return [f"    {c_type} {name} = ({c_type}) luaL_checkstring(L, {index});"]

    if type_info.kind == "pointer":
        expr = render_pointer_read_expr(c_type, str(index), name)
        return [f"    {c_type} {name} = {expr};"]

    if type_info.kind == "record":
        helper = f"lua_ggml_check_record_{record_helper_suffix(type_info.record_name or 'record')}"
        return [f"    {c_type} {name} = {helper}(L, {index});"]

    raise ValueError(f"Unsupported parameter mapping kind: {type_info.kind}")


def render_push_value(type_info: TypeInfo, value_expr: str) -> List[str]:
    if type_info.kind == "pointer":
        return [
            f"    if ({value_expr} == NULL) {{",
            "        lua_pushnil(L);",
            "    } else {",
            f"        lua_pushlightuserdata(L, (void *) {value_expr});",
            "    }",
        ]

    if type_info.kind in {"integer", "enum"}:
        return [f"    lua_pushinteger(L, (lua_Integer) {value_expr});"]

    if type_info.kind == "number":
        return [f"    lua_pushnumber(L, (lua_Number) {value_expr});"]

    if type_info.kind == "bool":
        return [f"    lua_pushboolean(L, ({value_expr}) ? 1 : 0);"]

    if type_info.kind == "string":
        return [
            f"    if ({value_expr} == NULL) {{",
            "        lua_pushnil(L);",
            "    } else {",
            f"        lua_pushstring(L, {value_expr});",
            "    }",
        ]

    if type_info.kind == "record":
        helper = f"lua_ggml_push_record_{record_helper_suffix(type_info.record_name or 'record')}"
        return [f"    {helper}(L, {value_expr});"]

    raise ValueError(f"Unsupported push mapping kind: {type_info.kind}")


def render_return_push(type_info: TypeInfo, value_name: str) -> List[str]:
    if type_info.kind == "void":
        return ["    return 0;"]

    lines = render_push_value(type_info, value_name)
    lines.append("    return 1;")
    return lines


def render_record_reader(record: RecordInfo) -> str:
    suffix = record_helper_suffix(record.name)
    shorthand_field = STRUCT_INTEGER_SHORTHANDS.get(record.name)
    lines = [
        f"static {record.c_spelling} lua_ggml_check_record_{suffix}(lua_State *L, int index) {{",
        "    int abs_index = lua_absindex(L, index);",
        f"    {record.c_spelling} value = ({record.c_spelling}) {{0}};",
        "",
    ]

    if shorthand_field:
        shorthand_type = next(field.type_info.c_spelling for field in record.fields if field.name == shorthand_field)
        lines.extend(
            [
                "    if (lua_isinteger(L, abs_index)) {",
                f"        value.{shorthand_field} = ({shorthand_type}) lua_tointeger(L, abs_index);",
                "        return value;",
                "    }",
                "",
            ]
        )

    lines.append("    luaL_checktype(L, abs_index, LUA_TTABLE);")
    lines.append("")

    for field in record.fields:
        field_label = f"{record.name}.{field.name}"
        lines.append(f"    lua_getfield(L, abs_index, \"{field.name}\");")
        lines.append("    if (!lua_isnil(L, -1)) {")

        if field.type_info.kind in {"integer", "enum"}:
            lines.append(
                f"        value.{field.name} = ({field.type_info.c_spelling}) luaL_checkinteger(L, -1);"
            )
        elif field.type_info.kind == "number":
            lines.append(
                f"        value.{field.name} = ({field.type_info.c_spelling}) luaL_checknumber(L, -1);"
            )
        elif field.type_info.kind == "bool":
            lines.append(
                f"        value.{field.name} = ({field.type_info.c_spelling}) (lua_toboolean(L, -1) != 0);"
            )
        elif field.type_info.kind == "string":
            lines.append(
                f"        value.{field.name} = ({field.type_info.c_spelling}) luaL_checkstring(L, -1);"
            )
        elif field.type_info.kind == "pointer":
            expr = render_pointer_read_expr(field.type_info.c_spelling, "-1", field_label)
            lines.append(f"        value.{field.name} = {expr};")
        elif field.type_info.kind == "record":
            nested_suffix = record_helper_suffix(field.type_info.record_name or "record")
            lines.append(
                f"        value.{field.name} = lua_ggml_check_record_{nested_suffix}(L, -1);"
            )
        else:
            raise ValueError(f"Unsupported record field kind: {field.type_info.kind}")

        lines.append("    }")
        lines.append("    lua_pop(L, 1);")
        lines.append("")

    lines.append("    return value;")
    lines.append("}")
    return "\n".join(lines)


def render_record_pusher(record: RecordInfo) -> str:
    suffix = record_helper_suffix(record.name)
    lines = [
        f"static void lua_ggml_push_record_{suffix}(lua_State *L, {record.c_spelling} value) {{",
        f"    lua_createtable(L, 0, {len(record.fields)});",
    ]

    for field in record.fields:
        lines.extend(render_push_value(field.type_info, f"value.{field.name}"))
        lines.append(f"    lua_setfield(L, -2, \"{field.name}\");")

    lines.append("}")
    return "\n".join(lines)


def render_wrapper(fn: FuncInfo) -> str:
    lines: List[str] = []
    wrapper_name = f"l_auto_{fn.c_name}"
    lines.append(f"static int {wrapper_name}(lua_State *L) {{")

    for i, param in enumerate(fn.params, start=1):
        lines.extend(render_param_read_expr(param.type_info, param.name, i))

    call_args = ", ".join(param.name for param in fn.params)
    if fn.return_type.kind == "void":
        lines.append(f"    {fn.c_name}({call_args});")
        lines.extend(render_return_push(fn.return_type, ""))
    else:
        lines.append(f"    {fn.return_type.c_spelling} result = {fn.c_name}({call_args});")
        lines.extend(render_return_push(fn.return_type, "result"))

    lines.append("}")
    return "\n".join(lines)


def render_c(
    functions: Sequence[FuncInfo],
    constants: Sequence[ConstantInfo],
    constant_alias_counts: Dict[str, int],
    records: Dict[str, RecordInfo],
    header_basenames: Sequence[str],
    header_basename: str,
) -> str:
    out: List[str] = []
    unique_headers: List[str] = []
    seen_headers: set[str] = set()
    for header_name in header_basenames:
        if header_name in seen_headers:
            continue
        seen_headers.add(header_name)
        unique_headers.append(header_name)

    out.append("/* Auto-generated by tools/gen_lua_bindings.py. Do not edit manually. */")
    out.append("#include <lua.h>")
    out.append("#include <lauxlib.h>")
    for header_name in unique_headers:
        out.append(f"#include \"{header_name}\"")
    out.append(f"#include \"{header_basename}\"")
    out.append("")
    out.extend(
        [
            "static void * lua_ggml_to_pointer(lua_State *L, int index, const char *name) {",
            "    if (lua_isnil(L, index)) {",
            "        return NULL;",
            "    }",
            "",
            "    if (!lua_islightuserdata(L, index) && !lua_isuserdata(L, index)) {",
            "        luaL_error(L, \"%s must be userdata, lightuserdata, or nil\", name);",
            "        return NULL;",
            "    }",
            "",
            "    return lua_touserdata(L, index);",
            "}",
            "",
        ]
    )

    for record_name in sorted(records.keys()):
        record = records[record_name]
        out.append(render_record_reader(record))
        out.append("")
        out.append(render_record_pusher(record))
        out.append("")

    for fn in functions:
        out.append(render_wrapper(fn))
        out.append("")

    out.append("static const luaL_Reg ggml_generated_funcs[] = {")
    for fn in functions:
        out.append(f"    {{\"{fn.lua_name}\", l_auto_{fn.c_name}}},")
    out.append("    {NULL, NULL}")
    out.append("};")
    out.append("")
    out.append("void lua_ggml_add_generated(lua_State *L) {")
    out.append("    luaL_setfuncs(L, ggml_generated_funcs, 0);")
    out.append("")
    for constant in constants:
        out.append(f"    lua_pushinteger(L, (lua_Integer) {constant.c_name});")
        out.append(f"    lua_setfield(L, -2, \"{constant.c_name}\");")
        if constant.lua_name != constant.c_name and constant_alias_counts.get(constant.lua_name, 0) == 1:
            out.append(f"    lua_pushinteger(L, (lua_Integer) {constant.c_name});")
            out.append(f"    lua_setfield(L, -2, \"{constant.lua_name}\");")
    out.append("}")
    out.append("")

    return "\n".join(out)


def render_h() -> str:
    return "\n".join(
        [
            "/* Auto-generated by tools/gen_lua_bindings.py. Do not edit manually. */",
            "#ifndef LUA_GGML_BINDINGS_H",
            "#define LUA_GGML_BINDINGS_H",
            "",
            "#include <lua.h>",
            "",
            "#ifdef __cplusplus",
            "extern \"C\" {",
            "#endif",
            "",
            "void lua_ggml_add_generated(lua_State *L);",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif",
            "",
        ]
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_file(path: str, content: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def load_clang_json_ast(header: str, clang_bin: str, extra_args: Sequence[str]) -> dict:
    cmd = [
        clang_bin,
        "-x",
        "c",
        "-std=c11",
        f"-I{os.path.dirname(header)}",
        "-Xclang",
        "-ast-dump=json",
        "-fsyntax-only",
    ]
    cmd.extend(extra_args)
    cmd.append(header)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"clang AST dump failed for {header} with exit code {proc.returncode}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse clang JSON AST output for {header}") from exc


def main() -> int:
    args = parse_args()

    headers = [normalize_path(header) for header in args.header]
    nm_libraries = [normalize_path(library) for library in args.nm_library]
    out_c = normalize_path(args.out_c)
    out_h = normalize_path(args.out_h)

    for header in headers:
        if not os.path.exists(header):
            print(f"error: header not found: {header}", file=sys.stderr)
            return 1

    excluded = set(EXCLUDED_BY_DEFAULT)
    excluded.update(args.exclude)

    try:
        ast_roots = [load_clang_json_ast(header, args.clang_bin, args.clang_arg) for header in headers]
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    typedefs = collect_typedefs(ast_roots)
    exported_symbols = collect_exported_symbols(nm_libraries) if nm_libraries else set()

    record_candidates = collect_record_candidates(ast_roots, typedefs)
    records = resolve_records(record_candidates)
    functions, skipped = find_functions(ast_roots, args.prefix, excluded, records, typedefs, exported_symbols)
    constants, constant_alias_counts = find_constants(ast_roots)

    c_content = render_c(
        functions,
        constants,
        constant_alias_counts,
        records,
        [os.path.basename(header) for header in headers],
        os.path.basename(out_h),
    )
    h_content = render_h()

    write_file(out_c, c_content)
    write_file(out_h, h_content)

    skipped_total = sum(skipped.values())
    print(f"Generated {len(functions)} binding wrappers")
    print(f"Generated {len(constants)} enum constants")
    print(f"Generated {len(records)} record helpers")
    print(f"Skipped {skipped_total} functions")
    for reason in sorted(skipped.keys()):
        print(f"  - {reason}: {skipped[reason]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
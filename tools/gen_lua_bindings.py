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
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EXCLUDED_BY_DEFAULT: set[str] = set()

STRUCT_INTEGER_SHORTHANDS = {
    # Preserve the existing user-facing API: ggml.init(mem_size)
    "ggml_init_params": "mem_size",
}

POINTER_REF_TYPES = {
    "void **": "void *",
    "struct ggml_context **": "struct ggml_context *",
}

POINTER_ARRAY_TYPES = {
    "const char **": ("string_array", "const char *"),
    "struct ggml_tensor **": ("pointer_array", "struct ggml_tensor *"),
    "const struct ggml_tensor * const *": (
        "pointer_array",
        "const struct ggml_tensor *",
    ),
    "ggml_backend_t *": ("pointer_array", "ggml_backend_t"),
    "ggml_backend_buffer_type_t *": ("pointer_array", "ggml_backend_buffer_type_t"),
}

RECORD_POINTER_TYPES = {
    "struct ggml_threadpool_params *": "ggml_threadpool_params",
    "const struct ggml_threadpool_params *": "ggml_threadpool_params",
}

ALLOW_NIL_ARRAY_PARAMS = {
    ("ggml_backend_sched_new", "bufts"),
    ("ggml_build_backward_expand", "grad_accs"),
}

ARRAY_COUNT_PARAMS = {
    ("ggml_backend_sched_new", "backends"): "n_backends",
    ("ggml_backend_sched_new", "bufts"): "n_backends",
    ("ggml_build_forward_select", "tensors"): "n_tensors",
    ("ggml_backend_compare_graph_backend", "test_nodes"): "num_test_nodes",
    ("ggml_gallocr_new_n", "bufts"): "n_bufs",
    ("gguf_set_arr_str", "data"): "n",
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

OWNED_POINTER_RETURNS = {
    "ggml_init",
    "ggml_backend_alloc_buffer",
    "ggml_backend_alloc_ctx_tensors",
    "ggml_backend_alloc_ctx_tensors_from_buft",
    "ggml_backend_buft_alloc_buffer",
    "ggml_backend_cpu_buffer_from_ptr",
    "ggml_backend_cpu_init",
    "ggml_backend_dev_init",
    "ggml_backend_event_new",
    "ggml_backend_init_best",
    "ggml_backend_init_by_name",
    "ggml_backend_init_by_type",
    "ggml_backend_sched_new",
    "ggml_gallocr_new",
    "ggml_gallocr_new_n",
    "ggml_threadpool_new",
    "gguf_init_empty",
    "gguf_init_from_file",
}


@dataclass(frozen=True)
class TypeInfo:
    kind: str
    c_spelling: str
    record_name: Optional[str] = None
    pointee_c_spelling: Optional[str] = None
    element_type: Optional["TypeInfo"] = None
    array_length_expr: Optional[str] = None


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
    parser = argparse.ArgumentParser(
        description="Generate Lua bindings for ggml using clang JSON AST"
    )
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
        action="append",
        default=[],
        help="Function prefix to include from headers (repeatable)",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Function prefix to strip from the exported Lua name (repeatable)",
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


def longest_matching_prefix(name: str, prefixes: Sequence[str]) -> Optional[str]:
    matches = [prefix for prefix in prefixes if name.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=len)


def lua_name_for_symbol(c_name: str, strip_prefixes: Sequence[str]) -> str:
    strip_prefix = longest_matching_prefix(c_name, strip_prefixes)
    if strip_prefix is None:
        return c_name
    lua_name = c_name[len(strip_prefix) :]
    return lua_name or c_name


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


def parse_fixed_array_type(type_text: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^(.*)\[([^\]]+)\]$", type_text)
    if match is None:
        return None
    return canonical_text(match.group(1)), canonical_text(match.group(2))


def strip_one_pointer(type_text: str) -> Optional[str]:
    stripped = canonical_text(type_text)
    if not stripped.endswith("*"):
        return None
    return canonical_text(stripped[:-1])


def extract_record_name(type_text: str) -> Optional[str]:
    normalized = canonical_text(type_text)
    normalized = re.sub(r"\bconst\b", "", normalized)
    normalized = canonical_text(normalized)
    if normalized.startswith("struct "):
        return normalized.split(" ", 1)[1]
    return None


def walk_nodes(node: dict) -> Iterable[dict]:
    yield node
    for child in node.get("inner", []):
        yield from walk_nodes(child)


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


def classify_type(
    qual: str, desugared: str, typedefs: Dict[str, str]
) -> Optional[TypeInfo]:
    qual_clean = canonical_text(qual)
    desugared_clean = canonical_text(desugared) if desugared else qual_clean
    qual_resolved = resolve_alias_text(qual_clean, typedefs)
    desugared_resolved = resolve_alias_text(desugared_clean, typedefs)

    if not qual_clean:
        return None

    for candidate in (qual_resolved, desugared_resolved, qual_clean, desugared_clean):
        array_parts = parse_fixed_array_type(candidate)
        if array_parts is None:
            continue

        element_text, length_expr = array_parts
        element_type = classify_type(element_text, element_text, typedefs)
        if element_type is None or element_type.kind in {
            "fixed_array",
            "pointer_ref",
            "pointer_array",
            "string_array",
            "record_pointer",
        }:
            return None
        return TypeInfo(
            "fixed_array",
            qual_clean,
            element_type=element_type,
            array_length_expr=length_expr,
        )

    if qual_clean == "void":
        return TypeInfo("void", qual_clean)

    if desugared_resolved == "_Bool" or qual_clean == "bool":
        return TypeInfo("bool", qual_clean)

    if qual_clean.startswith("enum ") or desugared_resolved.startswith("enum "):
        return TypeInfo("enum", qual_clean)

    if "(*" in desugared_resolved or "(*" in qual_resolved:
        return None

    if "*" in desugared_resolved or "*" in qual_resolved:
        for candidate in (
            qual_resolved,
            desugared_resolved,
            qual_clean,
            desugared_clean,
        ):
            mapping = POINTER_ARRAY_TYPES.get(candidate)
            if mapping is not None:
                kind, pointee = mapping
                return TypeInfo(kind, qual_clean, pointee_c_spelling=pointee)

        star_count = (
            desugared_resolved.count("*")
            if "*" in desugared_resolved
            else qual_resolved.count("*")
        )
        if star_count > 1:
            if qual_resolved in POINTER_REF_TYPES:
                return TypeInfo(
                    "pointer_ref",
                    qual_clean,
                    pointee_c_spelling=POINTER_REF_TYPES[qual_resolved],
                )
            if desugared_resolved in POINTER_REF_TYPES:
                return TypeInfo(
                    "pointer_ref",
                    qual_clean,
                    pointee_c_spelling=POINTER_REF_TYPES[desugared_resolved],
                )
            return None

        for candidate in (
            qual_resolved,
            desugared_resolved,
            qual_clean,
            desugared_clean,
        ):
            record_name = RECORD_POINTER_TYPES.get(candidate)
            if record_name is not None:
                return TypeInfo("record_pointer", qual_clean, record_name=record_name)

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


def has_deprecated_attr(node: dict) -> bool:
    return any(child.get("kind") == "DeprecatedAttr" for child in node.get("inner", []))


def get_param_name(param_node: dict, index: int) -> str:
    if name := param_node.get("name", ""):
        return name
    return f"arg{index}"


def analyze_record_candidate(
    node: dict, typedefs: Dict[str, str]
) -> Optional[RecordInfo]:
    name = node.get("name", "")
    if not name:
        return None

    field_nodes = [
        child for child in node.get("inner", []) if child.get("kind") == "FieldDecl"
    ]
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


def collect_record_candidates(
    ast_roots: Sequence[dict], typedefs: Dict[str, str]
) -> Dict[str, RecordInfo]:
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
    supported_field_kinds = {
        "integer",
        "enum",
        "number",
        "bool",
        "string",
        "pointer",
        "pointer_ref",
        "record",
        "fixed_array",
    }

    while progress:
        progress = False
        for name, record in candidates.items():
            if name in supported:
                continue

            unresolved = False
            for field in record.fields:
                if field.type_info.kind not in supported_field_kinds:
                    unresolved = True
                    break
                if (
                    field.type_info.kind == "record"
                    and field.type_info.record_name not in supported
                ):
                    unresolved = True
                    break

            if unresolved:
                continue

            supported[name] = record
            progress = True

    return supported


def analyze_function(
    fn_node: dict,
    prefixes: Sequence[str],
    strip_prefixes: Sequence[str],
    excluded: set[str],
    records: Dict[str, RecordInfo],
    typedefs: Dict[str, str],
) -> Tuple[Optional[FuncInfo], Optional[str]]:
    c_name = fn_node.get("name", "")
    if not c_name or longest_matching_prefix(c_name, prefixes) is None:
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
    if ret.kind in {"fixed_array", "pointer_array", "string_array", "record_pointer"}:
        return None, f"unsupported return type '{ret_qual}'"
    if ret.kind == "record" and ret.record_name not in records:
        return None, f"unsupported return type '{ret_qual}'"

    params: List[ParamInfo] = []
    param_nodes = [
        child
        for child in fn_node.get("inner", [])
        if child.get("kind") == "ParmVarDecl"
    ]
    for i, param_node in enumerate(param_nodes, start=1):
        ptype = param_node.get("type", {})
        p_qual = canonical_text(ptype.get("qualType", ""))
        p_desugared = canonical_text(ptype.get("desugaredQualType", p_qual))
        mapped = classify_type(p_qual, p_desugared, typedefs)
        if mapped is None:
            return None, f"unsupported param type '{p_qual}'"
        if mapped.kind == "record" and mapped.record_name not in records:
            return None, f"unsupported param type '{p_qual}'"
        if mapped.kind == "record_pointer" and mapped.record_name not in records:
            return None, f"unsupported param type '{p_qual}'"
        if (
            mapped.kind in {"pointer_array", "string_array"}
            and (c_name, get_param_name(param_node, i)) not in ARRAY_COUNT_PARAMS
        ):
            return None, f"unsupported param type '{p_qual}'"
        params.append(ParamInfo(get_param_name(param_node, i), mapped))

    lua_name = lua_name_for_symbol(c_name, strip_prefixes)
    return FuncInfo(c_name, lua_name, ret, tuple(params)), None


def find_functions(
    ast_roots: Sequence[dict],
    prefixes: Sequence[str],
    strip_prefixes: Sequence[str],
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

            info, reason = analyze_function(
                node, prefixes, strip_prefixes, excluded, records, typedefs
            )
            if info is None:
                if reason and c_name and c_name not in seen_skipped:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    seen_skipped.add(c_name)
                continue

            if exported_symbols and info.c_name not in exported_symbols:
                if info.c_name not in seen_skipped:
                    skipped["missing exported symbol"] = (
                        skipped.get("missing exported symbol", 0) + 1
                    )
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


def find_constants(
    ast_roots: Sequence[dict],
) -> Tuple[List[ConstantInfo], Dict[str, int]]:
    constants: Dict[str, ConstantInfo] = {}

    for ast_root in ast_roots:
        for node in walk_nodes(ast_root):
            if node.get("kind") != "EnumConstantDecl":
                continue

            c_name = node.get("name", "")
            if (
                not (c_name.startswith("GGML_") or c_name.startswith("GGUF_"))
                or c_name in constants
            ):
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
    return f'({c_type}) lua_ggml_to_pointer(L, {index_expr}, "{label}")'


def extra_indent(lines: Sequence[str], levels: int = 1) -> List[str]:
    prefix = "    " * levels
    return [prefix + line if line else "" for line in lines]


def render_stack_value_assignment(
    type_info: TypeInfo, target_expr: str, stack_index_expr: str, label: str
) -> List[str]:
    c_type = type_info.c_spelling

    if type_info.kind in {"integer", "enum"}:
        return [f"{target_expr} = ({c_type}) luaL_checkinteger(L, {stack_index_expr});"]

    if type_info.kind == "number":
        return [f"{target_expr} = ({c_type}) luaL_checknumber(L, {stack_index_expr});"]

    if type_info.kind == "bool":
        return [
            f"{target_expr} = ({c_type}) (lua_toboolean(L, {stack_index_expr}) != 0);"
        ]

    if type_info.kind == "string":
        return [f"{target_expr} = ({c_type}) luaL_checkstring(L, {stack_index_expr});"]

    if type_info.kind == "pointer":
        expr = render_pointer_read_expr(c_type, stack_index_expr, label)
        return [f"{target_expr} = {expr};"]

    if type_info.kind == "record":
        helper = f"lua_ggml_check_record_{record_helper_suffix(type_info.record_name or 'record')}"
        return [f"{target_expr} = {helper}(L, {stack_index_expr});"]

    raise ValueError(f"Unsupported stack assignment kind: {type_info.kind}")


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

    if type_info.kind == "pointer_ref":
        pointee = type_info.pointee_c_spelling or "void *"
        return [
            f'    {c_type} {name} = ({c_type}) lua_ggml_to_pointer_ref(L, {index}, "{name}", "{pointee}");'
        ]

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
            f'        lua_ggml_push_pointer(L, (void *) {value_expr}, "{type_info.c_spelling}", false);',
            "    }",
        ]

    if type_info.kind == "pointer_ref":
        pointee = type_info.pointee_c_spelling or "void *"
        return [
            f"    if ({value_expr} == NULL) {{",
            f'        lua_ggml_push_pointer_ref(L, NULL, "{pointee}");',
            "    } else {",
            f'        lua_ggml_push_pointer_ref(L, *((void **) {value_expr}), "{pointee}");',
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


def render_return_push(
    type_info: TypeInfo, value_name: str, function_name: str
) -> List[str]:
    if type_info.kind == "void":
        return ["    return 0;"]

    if type_info.kind == "pointer":
        owned = "true" if function_name in OWNED_POINTER_RETURNS else "false"
        return [
            f"    if ({value_name} == NULL) {{",
            "        lua_pushnil(L);",
            "    } else {",
            f'        lua_ggml_push_pointer(L, (void *) {value_name}, "{type_info.c_spelling}", {owned});',
            "    }",
            "    return 1;",
        ]

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
        shorthand_type = next(
            field.type_info.c_spelling
            for field in record.fields
            if field.name == shorthand_field
        )
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
        lines.append(f'    lua_getfield(L, abs_index, "{field.name}");')
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
            expr = render_pointer_read_expr(
                field.type_info.c_spelling, "-1", field_label
            )
            lines.append(f"        value.{field.name} = {expr};")
        elif field.type_info.kind == "pointer_ref":
            pointee = field.type_info.pointee_c_spelling or "void *"
            lines.append(
                f'        value.{field.name} = ({field.type_info.c_spelling}) lua_ggml_to_pointer_ref(L, -1, "{field_label}", "{pointee}");'
            )
        elif field.type_info.kind == "fixed_array":
            element_type = field.type_info.element_type
            loop_var = f"i_{field.name}"
            size_expr = f"(sizeof(value.{field.name}) / sizeof(value.{field.name}[0]))"
            if element_type is None:
                raise ValueError(
                    f"Fixed array field missing element type: {field_label}"
                )
            lines.append("        luaL_checktype(L, -1, LUA_TTABLE);")
            lines.append(
                f"        for (size_t {loop_var} = 0; {loop_var} < {size_expr}; ++{loop_var}) {{"
            )
            lines.append(
                f"            lua_rawgeti(L, -1, (lua_Integer) {loop_var} + 1);"
            )
            lines.append("            if (!lua_isnil(L, -1)) {")
            lines.extend(
                extra_indent(
                    render_stack_value_assignment(
                        element_type,
                        f"value.{field.name}[{loop_var}]",
                        "-1",
                        f"{field_label}[{loop_var}]",
                    ),
                    3,
                )
            )
            lines.append("            }")
            lines.append("            lua_pop(L, 1);")
            lines.append("        }")
        elif field.type_info.kind == "record":
            nested_suffix = record_helper_suffix(
                field.type_info.record_name or "record"
            )
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


def render_record_filler(record: RecordInfo) -> str:
    suffix = record_helper_suffix(record.name)
    lines = [
        f"static void lua_ggml_fill_record_{suffix}(lua_State *L, int index, {record.c_spelling} value) {{",
        "    int abs_index = lua_absindex(L, index);",
    ]

    for field in record.fields:
        if field.type_info.kind == "fixed_array":
            element_type = field.type_info.element_type
            loop_var = f"i_{field.name}"
            size_expr = f"(sizeof(value.{field.name}) / sizeof(value.{field.name}[0]))"
            if element_type is None:
                raise ValueError(
                    f"Fixed array field missing element type: {record.name}.{field.name}"
                )
            lines.append(f"    lua_createtable(L, (int) {size_expr}, 0);")
            lines.append(
                f"    for (size_t {loop_var} = 0; {loop_var} < {size_expr}; ++{loop_var}) {{"
            )
            lines.extend(
                extra_indent(
                    render_push_value(element_type, f"value.{field.name}[{loop_var}]"),
                    1,
                )
            )
            lines.append(f"        lua_rawseti(L, -2, (lua_Integer) {loop_var} + 1);")
            lines.append("    }")
            lines.append(f'    lua_setfield(L, abs_index, "{field.name}");')
            continue

        lines.extend(render_push_value(field.type_info, f"value.{field.name}"))
        lines.append(f'    lua_setfield(L, abs_index, "{field.name}");')

    lines.append("}")
    return "\n".join(lines)


def render_record_pusher(record: RecordInfo) -> str:
    suffix = record_helper_suffix(record.name)
    lines = [
        f"static void lua_ggml_push_record_{suffix}(lua_State *L, {record.c_spelling} value) {{",
        f"    lua_createtable(L, 0, {len(record.fields)});",
        f"    lua_ggml_fill_record_{suffix}(L, -1, value);",
    ]

    lines.append("}")
    return "\n".join(lines)


def render_wrapper(fn: FuncInfo, records: Dict[str, RecordInfo]) -> str:
    lines: List[str] = []
    wrapper_name = f"l_auto_{fn.c_name}"
    count_param_names = {
        ARRAY_COUNT_PARAMS[(fn.c_name, param.name)]
        for param in fn.params
        if (fn.c_name, param.name) in ARRAY_COUNT_PARAMS
    }
    param_indices = {param.name: i for i, param in enumerate(fn.params, start=1)}

    lines.append(f"static int {wrapper_name}(lua_State *L) {{")

    for param in fn.params:
        if param.name not in count_param_names:
            continue
        lines.extend(
            render_param_read_expr(
                param.type_info, param.name, param_indices[param.name]
            )
        )

    for i, param in enumerate(fn.params, start=1):
        if param.name in count_param_names:
            continue

        if param.type_info.kind in {"pointer_array", "string_array"}:
            count_param_name = ARRAY_COUNT_PARAMS[(fn.c_name, param.name)]
            allow_nil = (fn.c_name, param.name) in ALLOW_NIL_ARRAY_PARAMS
            element_c_spelling = param.type_info.pointee_c_spelling
            if element_c_spelling is None:
                raise ValueError(
                    f"Missing array element type for {fn.c_name}.{param.name}"
                )

            lines.append(f"    {param.type_info.c_spelling} {param.name} = NULL;")
            lines.append(
                f"    size_t {param.name}_count = (size_t) {count_param_name};"
            )
            lines.append(f"    if (lua_isnil(L, {i})) {{")
            if allow_nil:
                lines.append(f"        {param.name} = NULL;")
            else:
                lines.append(
                    f'        return luaL_argerror(L, {i}, "{param.name} table expected");'
                )
            lines.append("    } else {")
            lines.append(f"        luaL_checktype(L, {i}, LUA_TTABLE);")
            lines.append(f"        lua_Integer {param.name}_len = luaL_len(L, {i});")
            lines.append(
                f"        if ((size_t) {param.name}_len != {param.name}_count) {{"
            )
            lines.append(
                f'            return luaL_error(L, "{param.name} length must match {count_param_name} (%lld expected, got %lld)", (long long) {param.name}_count, (long long) {param.name}_len);'
            )
            lines.append("        }")
            lines.append(
                f"        {param.name} = ({param.type_info.c_spelling}) lua_newuserdatauv(L, {param.name}_count * sizeof(*{param.name}), 0);"
            )
            lines.append(
                f"        for (size_t i_{param.name} = 0; i_{param.name} < {param.name}_count; ++i_{param.name}) {{"
            )
            lines.append(
                f"            lua_rawgeti(L, {i}, (lua_Integer) i_{param.name} + 1);"
            )
            if param.type_info.kind == "string_array":
                lines.append(
                    f"            {param.name}[i_{param.name}] = ({element_c_spelling}) luaL_checkstring(L, -1);"
                )
            else:
                if allow_nil:
                    lines.append("            if (lua_isnil(L, -1)) {")
                    lines.append(
                        f"                {param.name}[i_{param.name}] = NULL;"
                    )
                    lines.append("            } else {")
                    lines.append(
                        f'                {param.name}[i_{param.name}] = ({element_c_spelling}) lua_ggml_to_pointer(L, -1, "{param.name}");'
                    )
                    lines.append("            }")
                else:
                    lines.append(
                        f'            {param.name}[i_{param.name}] = ({element_c_spelling}) lua_ggml_to_pointer(L, -1, "{param.name}");'
                    )
            lines.append("            lua_pop(L, 1);")
            lines.append("        }")
            lines.append("    }")
            continue

        if param.type_info.kind == "record_pointer":
            record_name = param.type_info.record_name or "record"
            record = records.get(record_name)
            if record is None:
                raise ValueError(f"Missing record info for {fn.c_name}.{param.name}")
            suffix = record_helper_suffix(record_name)
            lines.append(
                f"    {record.c_spelling} {param.name}_storage = ({record.c_spelling}) {{0}};"
            )
            lines.append(f"    {param.type_info.c_spelling} {param.name} = NULL;")
            lines.append(f"    int {param.name}_table_index = 0;")
            lines.append(f"    if (lua_istable(L, {i})) {{")
            lines.append(
                f"        {param.name}_storage = lua_ggml_check_record_{suffix}(L, {i});"
            )
            lines.append(f"        {param.name} = &{param.name}_storage;")
            lines.append(f"        {param.name}_table_index = lua_absindex(L, {i});")
            lines.append("    } else {")
            lines.append(
                f'        {param.name} = ({param.type_info.c_spelling}) lua_ggml_to_pointer(L, {i}, "{param.name}");'
            )
            lines.append("    }")
            continue

        lines.extend(render_param_read_expr(param.type_info, param.name, i))

    call_args = ", ".join(param.name for param in fn.params)
    if fn.return_type.kind == "void":
        lines.append(f"    {fn.c_name}({call_args});")
        for param in fn.params:
            if param.type_info.kind != "record_pointer":
                continue
            if param.type_info.c_spelling.startswith("const "):
                continue
            record_name = param.type_info.record_name or "record"
            suffix = record_helper_suffix(record_name)
            lines.append(f"    if ({param.name}_table_index != 0) {{")
            lines.append(
                f"        lua_ggml_fill_record_{suffix}(L, {param.name}_table_index, *{param.name});"
            )
            lines.append("    }")
        lines.extend(render_return_push(fn.return_type, "", fn.c_name))
    else:
        lines.append(
            f"    {fn.return_type.c_spelling} result = {fn.c_name}({call_args});"
        )
        for param in fn.params:
            if param.type_info.kind != "record_pointer":
                continue
            if param.type_info.c_spelling.startswith("const "):
                continue
            record_name = param.type_info.record_name or "record"
            suffix = record_helper_suffix(record_name)
            lines.append(f"    if ({param.name}_table_index != 0) {{")
            lines.append(
                f"        lua_ggml_fill_record_{suffix}(L, {param.name}_table_index, *{param.name});"
            )
            lines.append("    }")
        lines.extend(render_return_push(fn.return_type, "result", fn.c_name))

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

    out.append(
        "/* Auto-generated by tools/gen_lua_bindings.py. Do not edit manually. */"
    )
    out.append("#include <lua.h>")
    out.append("#include <lauxlib.h>")
    for header_name in unique_headers:
        out.append(f'#include "{header_name}"')
    out.append('#include "lua_ggml_support.h"')
    out.append(f'#include "{header_basename}"')
    out.append("")

    for record_name in sorted(records.keys()):
        record = records[record_name]
        out.append(render_record_reader(record))
        out.append("")
        out.append(render_record_filler(record))
        out.append("")
        out.append(render_record_pusher(record))
        out.append("")

    for fn in functions:
        out.append(render_wrapper(fn, records))
        out.append("")

    out.append("static const luaL_Reg ggml_generated_funcs[] = {")
    for fn in functions:
        out.append(f'    {{"{fn.lua_name}", l_auto_{fn.c_name}}},')
    out.append("    {NULL, NULL}")
    out.append("};")
    out.append("")
    out.append("void lua_ggml_add_generated(lua_State *L) {")
    out.append("    luaL_setfuncs(L, ggml_generated_funcs, 0);")
    out.append("")
    for constant in constants:
        out.append(f"    lua_pushinteger(L, (lua_Integer) {constant.c_name});")
        out.append(f'    lua_setfield(L, -2, "{constant.c_name}");')
        if (
            constant.lua_name != constant.c_name
            and constant_alias_counts.get(constant.lua_name, 0) == 1
        ):
            out.append(f"    lua_pushinteger(L, (lua_Integer) {constant.c_name});")
            out.append(f'    lua_setfield(L, -2, "{constant.lua_name}");')
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
            'extern "C" {',
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
    if parent := os.path.dirname(path):
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
        raise RuntimeError(
            f"clang AST dump failed for {header} with exit code {proc.returncode}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse clang JSON AST output for {header}"
        ) from exc


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
    prefixes = args.prefix or ["ggml_"]
    strip_prefixes = args.strip_prefix or ["ggml_"]

    try:
        ast_roots = [
            load_clang_json_ast(header, args.clang_bin, args.clang_arg)
            for header in headers
        ]
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    typedefs = collect_typedefs(ast_roots)
    exported_symbols = collect_exported_symbols(nm_libraries) if nm_libraries else set()

    record_candidates = collect_record_candidates(ast_roots, typedefs)
    records = resolve_records(record_candidates)
    functions, skipped = find_functions(
        ast_roots,
        prefixes,
        strip_prefixes,
        excluded,
        records,
        typedefs,
        exported_symbols,
    )
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

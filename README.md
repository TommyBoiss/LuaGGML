# LuaGGML

Lua bindings for [ggml](./ggml/) built as a Lua C module (`ggml.so`).

## Status

This project currently uses `lightuserdata`/raw pointers for `ggml_context*` and `ggml_tensor*`.
It is fast and minimal, but does not yet provide high-level Lua memory ownership helpers.

## Requirements

- CMake >= 3.14
- C/C++ compiler toolchain
- Lua 5.4 runtime and development headers
- Python 3 (for wrapper generation)
- `clang` CLI (for JSON AST parsing in the binding generator)

## Build
```bash
git submodule update --init --recursive
```

```bash
cmake -S . -B build
cmake --build build -j
```

Outputs:
- `build/ggml.so` (Lua module)
- `build/ggml/src/libggml.so` (ggml shared library)
- `build/generated/lua_ggml_bindings.c` (auto-generated wrappers)


## Usage Example

```lua
package.cpath = "./build/?.so;" .. package.cpath

local ggml = require("ggml")

local ctx = ggml.init(16 * 1024 * 1024)
assert(ctx ~= nil)

-- Generated wrappers are also registered:
assert(type(ggml.permute) == "function")
```

## Binding Generation

Auto-generation is enabled by default through CMake option:

```bash
-DLUA_GGML_AUTO_BINDINGS=ON
```

Generator script:
- `tools/gen_lua_bindings.py`

The generator intentionally skips unsupported signatures (function pointers, pointer-to-pointer cases, variadics, and a few complex typedef-based APIs) and prints a skip summary during build.

It currently scans these headers:
- `ggml/include/ggml.h`
- `ggml/include/ggml-backend.h`
- `ggml/include/ggml-cpu.h`
  

## License

- This project: MIT (`LICENSE`)
- Vendored dependency: ggml under MIT (`ggml/LICENSE`)
- See `THIRD_PARTY_NOTICES.md` for details

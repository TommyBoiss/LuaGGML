# LuaGGML

Lua bindings for [ggml](./ggml/) built as a Lua C module (`ggml.so`).

## Status

This project now exposes ggml pointers as managed Lua userdata handles by default.

Current caveat:
- Lua callbacks are still intentionally blocked for ggml APIs that may invoke from worker threads (for example logging, abort hooks, scheduler eval callbacks, and custom ops). Calling back into a Lua state from ggml worker threads is not safe without a dedicated callback marshalling design.

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

## Verify

Run the integration tests:
```bash
lua tests/test.lua
lua examples/gpt2.lua --help
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
assert(ctx:is_owned())

-- Generated wrappers are also registered:
assert(type(ggml.permute) == "function")

local tensor = ggml.new_tensor_1d(ctx, ggml.TYPE_I32, 2)
tensor:set_data(string.pack("<i4i4", 10, 20))
assert(tensor:shape()[1] == 2)

ggml.free(ctx)
assert(ctx:is_null())
```

## Binding Generation

Auto-generation is enabled by default through CMake option:

```bash
-DLUA_GGML_AUTO_BINDINGS=ON
```

Generator script:
- `tools/gen_lua_bindings.py`

The generator intentionally skips signatures that are unsafe or not yet expressible through the current type mapper, and prints a skip summary during build.

It currently scans these headers:
- `ggml/include/ggml.h`
- `ggml/include/ggml-alloc.h`
- `ggml/include/ggml-backend.h`
- `ggml/include/ggml-cpu.h`
- `ggml/include/gguf.h`

And conditionally scans additional backend headers when their corresponding ggml backend targets are built, such as CUDA, Metal, Vulkan, OpenCL, SYCL, RPC, BLAS, and other optional backends.
  

## License

- This project: MIT (`LICENSE`)
- Vendored dependency: ggml under MIT (`ggml/LICENSE`)
- See `THIRD_PARTY_NOTICES.md` for details

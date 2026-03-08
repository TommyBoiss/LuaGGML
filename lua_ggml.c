#include <lua.h>
#include <lauxlib.h>

#if defined(LUA_GGML_HAS_GENERATED_BINDINGS) && LUA_GGML_HAS_GENERATED_BINDINGS
#include "lua_ggml_bindings.h"
#endif

static const luaL_Reg ggml_funcs[] = {
    {NULL, NULL}
};

int luaopen_ggml(lua_State *L) {
    luaL_newlib(L, ggml_funcs);

#if defined(LUA_GGML_HAS_GENERATED_BINDINGS) && LUA_GGML_HAS_GENERATED_BINDINGS
    lua_ggml_add_generated(L);
#endif

    return 1;
}
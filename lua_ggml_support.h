#ifndef LUA_GGML_SUPPORT_H
#define LUA_GGML_SUPPORT_H

#include <stdbool.h>

#include <lua.h>

#ifdef __cplusplus
extern "C" {
#endif

void * lua_ggml_to_pointer(lua_State *L, int index, const char *name);
void ** lua_ggml_to_pointer_ref(lua_State *L, int index, const char *name, const char *c_type);
void lua_ggml_push_pointer(lua_State *L, void *ptr, const char *c_type, bool owned);
void lua_ggml_push_pointer_ref(lua_State *L, void *ptr, const char *c_type);
void lua_ggml_register_metatables(lua_State *L);

#ifdef __cplusplus
}
#endif

#endif
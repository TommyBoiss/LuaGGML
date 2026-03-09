#include <lua.h>
#include <lauxlib.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "lua_ggml_support.h"

#if defined(LUA_GGML_HAS_GENERATED_BINDINGS) && LUA_GGML_HAS_GENERATED_BINDINGS
#include "lua_ggml_bindings.h"
#endif

typedef struct lua_ggml_handle {
    void * ptr;
    const char * c_type;
    unsigned char owned;
} lua_ggml_handle;

typedef struct lua_ggml_cplan_ud {
    struct ggml_cplan plan;
} lua_ggml_cplan_ud;

typedef struct lua_ggml_pointer_ref {
    void * ptr;
    const char * c_type;
} lua_ggml_pointer_ref;

static const char * LUA_GGML_HANDLE_MT = "ggml.handle";
static const char * LUA_GGML_CPLAN_MT = "ggml.cplan";
static const char * LUA_GGML_POINTER_REF_MT = "ggml.pointer_ref";

static bool lua_ggml_type_has(const char * c_type, const char * needle) {
    return c_type != NULL && strstr(c_type, needle) != NULL;
}

static lua_ggml_handle * lua_ggml_test_handle(lua_State *L, int index) {
    return (lua_ggml_handle *) luaL_testudata(L, index, LUA_GGML_HANDLE_MT);
}

static lua_ggml_handle * lua_ggml_check_handle(lua_State *L, int index) {
    return (lua_ggml_handle *) luaL_checkudata(L, index, LUA_GGML_HANDLE_MT);
}

static lua_ggml_cplan_ud * lua_ggml_test_cplan(lua_State *L, int index) {
    return (lua_ggml_cplan_ud *) luaL_testudata(L, index, LUA_GGML_CPLAN_MT);
}

static lua_ggml_cplan_ud * lua_ggml_check_cplan(lua_State *L, int index) {
    return (lua_ggml_cplan_ud *) luaL_checkudata(L, index, LUA_GGML_CPLAN_MT);
}

static lua_ggml_pointer_ref * lua_ggml_test_pointer_ref(lua_State *L, int index) {
    return (lua_ggml_pointer_ref *) luaL_testudata(L, index, LUA_GGML_POINTER_REF_MT);
}

static lua_ggml_pointer_ref * lua_ggml_check_pointer_ref(lua_State *L, int index) {
    return (lua_ggml_pointer_ref *) luaL_checkudata(L, index, LUA_GGML_POINTER_REF_MT);
}

static bool lua_ggml_handle_is_tensor(const lua_ggml_handle * handle) {
    return handle != NULL && lua_ggml_type_has(handle->c_type, "ggml_tensor");
}

static bool lua_ggml_handle_is_backend(const lua_ggml_handle * handle) {
    return handle != NULL && strcmp(handle->c_type, "ggml_backend_t") == 0;
}

static bool lua_ggml_handle_is_backend_buffer(const lua_ggml_handle * handle) {
    return handle != NULL && lua_ggml_type_has(handle->c_type, "ggml_backend_buffer");
}

static bool lua_ggml_handle_is_context(const lua_ggml_handle * handle) {
    return handle != NULL && lua_ggml_type_has(handle->c_type, "ggml_context");
}

static void lua_ggml_release_handle(lua_ggml_handle * handle) {
    if (handle == NULL || handle->ptr == NULL || !handle->owned) {
        return;
    }

    if (lua_ggml_type_has(handle->c_type, "ggml_backend_buffer")) {
        ggml_backend_buffer_free((ggml_backend_buffer_t) handle->ptr);
    } else if (strcmp(handle->c_type, "ggml_backend_t") == 0) {
        ggml_backend_free((ggml_backend_t) handle->ptr);
    } else if (strcmp(handle->c_type, "ggml_gallocr_t") == 0) {
        ggml_gallocr_free((ggml_gallocr_t) handle->ptr);
    } else if (strcmp(handle->c_type, "ggml_backend_sched_t") == 0) {
        ggml_backend_sched_free((ggml_backend_sched_t) handle->ptr);
    } else if (lua_ggml_type_has(handle->c_type, "ggml_threadpool")) {
        ggml_threadpool_free((struct ggml_threadpool *) handle->ptr);
    } else if (strcmp(handle->c_type, "ggml_backend_event_t") == 0) {
        ggml_backend_event_free((ggml_backend_event_t) handle->ptr);
    } else if (lua_ggml_type_has(handle->c_type, "gguf_context")) {
        gguf_free((struct gguf_context *) handle->ptr);
    } else if (lua_ggml_type_has(handle->c_type, "ggml_context")) {
        ggml_free((struct ggml_context *) handle->ptr);
    }

    handle->ptr = NULL;
    handle->owned = 0;
}

static void lua_ggml_invalidate_handle(lua_State *L, int index) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, index);
    if (handle != NULL) {
        handle->ptr = NULL;
        handle->owned = 0;
    }
}

void * lua_ggml_to_pointer(lua_State *L, int index, const char *name) {
    if (lua_isnoneornil(L, index)) {
        return NULL;
    }

    lua_ggml_handle * handle = lua_ggml_test_handle(L, index);
    if (handle != NULL) {
        return handle->ptr;
    }

    lua_ggml_cplan_ud * cplan = lua_ggml_test_cplan(L, index);
    if (cplan != NULL) {
        return &cplan->plan;
    }

    lua_ggml_pointer_ref * pointer_ref = lua_ggml_test_pointer_ref(L, index);
    if (pointer_ref != NULL) {
        return pointer_ref->ptr;
    }

    if (lua_islightuserdata(L, index)) {
        return lua_touserdata(L, index);
    }

    if (lua_isuserdata(L, index)) {
        return lua_touserdata(L, index);
    }

    luaL_error(L, "%s must be userdata, lightuserdata, or nil", name);
    return NULL;
}

void ** lua_ggml_to_pointer_ref(lua_State *L, int index, const char *name, const char *c_type) {
    lua_ggml_pointer_ref * pointer_ref;

    if (lua_isnoneornil(L, index)) {
        return NULL;
    }

    pointer_ref = lua_ggml_test_pointer_ref(L, index);
    if (pointer_ref == NULL) {
        luaL_error(L, "%s must be ggml.pointer_ref userdata or nil", name);
        return NULL;
    }

    if (c_type != NULL) {
        if (pointer_ref->c_type == NULL || strcmp(pointer_ref->c_type, "void *") == 0) {
            pointer_ref->c_type = c_type;
        } else if (strcmp(pointer_ref->c_type, c_type) != 0) {
            luaL_error(L, "%s pointer_ref type mismatch: have '%s', need '%s'", name, pointer_ref->c_type, c_type);
            return NULL;
        }
    }

    return &pointer_ref->ptr;
}

void lua_ggml_push_pointer(lua_State *L, void *ptr, const char *c_type, bool owned) {
    lua_ggml_handle * handle;

    if (ptr == NULL) {
        lua_pushnil(L);
        return;
    }

    handle = (lua_ggml_handle *) lua_newuserdata(L, sizeof(*handle));
    handle->ptr = ptr;
    handle->c_type = c_type != NULL ? c_type : "void *";
    handle->owned = owned ? 1 : 0;

    luaL_getmetatable(L, LUA_GGML_HANDLE_MT);
    lua_setmetatable(L, -2);
}

void lua_ggml_push_pointer_ref(lua_State *L, void *ptr, const char *c_type) {
    lua_ggml_pointer_ref * pointer_ref = (lua_ggml_pointer_ref *) lua_newuserdata(L, sizeof(*pointer_ref));
    pointer_ref->ptr = ptr;
    pointer_ref->c_type = c_type != NULL ? c_type : "void *";

    luaL_getmetatable(L, LUA_GGML_POINTER_REF_MT);
    lua_setmetatable(L, -2);
}

static const struct ggml_tensor * lua_ggml_check_tensor(lua_State *L, int index, const char *name) {
    const struct ggml_tensor * tensor = (const struct ggml_tensor *) lua_ggml_to_pointer(L, index, name);
    if (tensor == NULL) {
        luaL_error(L, "%s must not be nil", name);
    }
    return tensor;
}

static struct ggml_tensor * lua_ggml_check_tensor_mut(lua_State *L, int index, const char *name) {
    return (struct ggml_tensor *) lua_ggml_check_tensor(L, index, name);
}

static void ** lua_ggml_check_pointer_table(lua_State *L, int index, const char *name, size_t * count_out, bool allow_nil) {
    size_t count;
    size_t i;
    void ** items;

    index = lua_absindex(L, index);
    luaL_checktype(L, index, LUA_TTABLE);

    count = (size_t) lua_rawlen(L, index);
    items = count > 0 ? (void **) calloc(count, sizeof(*items)) : NULL;

    if (count > 0 && items == NULL) {
        luaL_error(L, "failed to allocate pointer array for %s", name);
    }

    for (i = 0; i < count; ++i) {
        lua_rawgeti(L, index, (lua_Integer) i + 1);
        if (lua_isnil(L, -1) && allow_nil) {
            items[i] = NULL;
        } else {
            items[i] = lua_ggml_to_pointer(L, -1, name);
            if (!allow_nil && items[i] == NULL) {
                free(items);
                luaL_error(L, "%s[%zu] must not be nil", name, i + 1);
            }
        }
        lua_pop(L, 1);
    }

    *count_out = count;
    return items;
}

static int lua_ggml_push_tensor_table(lua_State *L, struct ggml_tensor ** tensors, int n_tensors) {
    int i;
    lua_createtable(L, n_tensors, 0);
    for (i = 0; i < n_tensors; ++i) {
        lua_ggml_push_pointer(L, tensors[i], "struct ggml_tensor *", false);
        lua_rawseti(L, -2, i + 1);
    }
    return 1;
}

static int l_tensor_set_data(lua_State *L) {
    struct ggml_tensor * tensor = lua_ggml_check_tensor_mut(L, 1, "tensor");
    size_t data_len = 0;
    const char * data = luaL_checklstring(L, 2, &data_len);
    lua_Integer offset_i = luaL_optinteger(L, 3, 0);
    lua_Integer size_i = luaL_optinteger(L, 4, (lua_Integer) data_len);
    size_t offset;
    size_t size;
    size_t tensor_bytes;
    ggml_backend_buffer_t buf;

    if (offset_i < 0 || size_i < 0) {
        return luaL_error(L, "offset and size must be >= 0");
    }

    offset = (size_t) offset_i;
    size = (size_t) size_i;

    if (size > data_len) {
        return luaL_error(L, "size (%zu) exceeds byte-string length (%zu)", size, data_len);
    }

    tensor_bytes = ggml_nbytes(tensor);
    if (offset > tensor_bytes || size > tensor_bytes - offset) {
        return luaL_error(L, "out-of-bounds write: offset=%zu size=%zu tensor_bytes=%zu", offset, size, tensor_bytes);
    }

    buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf != NULL) {
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        if (tensor->data == NULL) {
            return luaL_error(L, "tensor has no data allocation");
        }
        memcpy((char *) tensor->data + offset, data, size);
    }

    return 0;
}

static int l_tensor_get_data(lua_State *L) {
    const struct ggml_tensor * tensor = lua_ggml_check_tensor(L, 1, "tensor");
    lua_Integer offset_i = luaL_optinteger(L, 2, 0);
    size_t tensor_bytes = ggml_nbytes(tensor);
    size_t default_size;
    lua_Integer size_i;
    size_t offset;
    size_t size;
    ggml_backend_buffer_t buf;
    luaL_Buffer b;
    char * out;

    if (offset_i < 0) {
        return luaL_error(L, "offset must be >= 0");
    }

    offset = (size_t) offset_i;
    if (offset > tensor_bytes) {
        return luaL_error(L, "offset (%zu) exceeds tensor bytes (%zu)", offset, tensor_bytes);
    }

    default_size = tensor_bytes - offset;
    size_i = luaL_optinteger(L, 3, (lua_Integer) default_size);
    if (size_i < 0) {
        return luaL_error(L, "size must be >= 0");
    }

    size = (size_t) size_i;
    if (size > tensor_bytes - offset) {
        return luaL_error(L, "out-of-bounds read: offset=%zu size=%zu tensor_bytes=%zu", offset, size, tensor_bytes);
    }

    luaL_buffinitsize(L, &b, size);
    out = luaL_prepbuffsize(&b, size);

    buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf != NULL) {
        ggml_backend_tensor_get(tensor, out, offset, size);
    } else {
        if (tensor->data == NULL) {
            return luaL_error(L, "tensor has no data allocation");
        }
        memcpy(out, (const char *) tensor->data + offset, size);
    }

    luaL_addsize(&b, size);
    luaL_pushresult(&b);
    return 1;
}

static int l_handle_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    bool freed = handle->ptr != NULL && handle->owned;
    lua_ggml_release_handle(handle);
    lua_pushboolean(L, freed ? 1 : 0);
    return 1;
}

static int l_handle_is_owned(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    lua_pushboolean(L, handle->owned ? 1 : 0);
    return 1;
}

static int l_handle_is_null(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    lua_pushboolean(L, handle->ptr == NULL ? 1 : 0);
    return 1;
}

static int l_handle_type_name(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    lua_pushstring(L, handle->c_type != NULL ? handle->c_type : "void *");
    return 1;
}

static int l_handle_ptr(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    if (handle->ptr == NULL) {
        lua_pushnil(L);
    } else {
        lua_pushlightuserdata(L, handle->ptr);
    }
    return 1;
}

static int l_handle_name(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    if (handle->ptr == NULL) {
        lua_pushnil(L);
        return 1;
    }

    if (lua_ggml_handle_is_tensor(handle)) {
        lua_pushstring(L, ggml_get_name((const struct ggml_tensor *) handle->ptr));
        return 1;
    }

    if (lua_ggml_handle_is_backend(handle)) {
        lua_pushstring(L, ggml_backend_name((ggml_backend_t) handle->ptr));
        return 1;
    }

    if (lua_ggml_handle_is_backend_buffer(handle)) {
        lua_pushstring(L, ggml_backend_buffer_name((ggml_backend_buffer_t) handle->ptr));
        return 1;
    }

    if (strcmp(handle->c_type, "ggml_backend_dev_t") == 0) {
        lua_pushstring(L, ggml_backend_dev_name((ggml_backend_dev_t) handle->ptr));
        return 1;
    }

    lua_pushnil(L);
    return 1;
}

static int l_handle_nbytes(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    if (!lua_ggml_handle_is_tensor(handle) || handle->ptr == NULL) {
        return luaL_error(L, "nbytes() is only available on ggml_tensor handles");
    }
    lua_pushinteger(L, (lua_Integer) ggml_nbytes((const struct ggml_tensor *) handle->ptr));
    return 1;
}

static int l_handle_nelements(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    if (!lua_ggml_handle_is_tensor(handle) || handle->ptr == NULL) {
        return luaL_error(L, "nelements() is only available on ggml_tensor handles");
    }
    lua_pushinteger(L, (lua_Integer) ggml_nelements((const struct ggml_tensor *) handle->ptr));
    return 1;
}

static int l_handle_shape(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    const struct ggml_tensor * tensor;
    int n_dims;
    int i;

    if (!lua_ggml_handle_is_tensor(handle) || handle->ptr == NULL) {
        return luaL_error(L, "shape() is only available on ggml_tensor handles");
    }

    tensor = (const struct ggml_tensor *) handle->ptr;
    n_dims = ggml_n_dims(tensor);
    lua_createtable(L, n_dims, 0);
    for (i = 0; i < n_dims; ++i) {
        lua_pushinteger(L, (lua_Integer) tensor->ne[i]);
        lua_rawseti(L, -2, i + 1);
    }
    return 1;
}

static int l_handle_gc(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    lua_ggml_release_handle(handle);
    return 0;
}

static int l_handle_tostring(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_check_handle(L, 1);
    lua_pushfstring(L, "%s: %p%s", handle->c_type != NULL ? handle->c_type : "void *", handle->ptr, handle->owned ? " [owned]" : "");
    return 1;
}

static int l_pointer_ref_get(lua_State *L) {
    lua_ggml_pointer_ref * pointer_ref = lua_ggml_check_pointer_ref(L, 1);
    bool owned = lua_toboolean(L, 2) != 0;
    lua_ggml_push_pointer(L, pointer_ref->ptr, pointer_ref->c_type != NULL ? pointer_ref->c_type : "void *", owned);
    return 1;
}

static int l_pointer_ref_set(lua_State *L) {
    lua_ggml_pointer_ref * pointer_ref = lua_ggml_check_pointer_ref(L, 1);
    pointer_ref->ptr = lua_ggml_to_pointer(L, 2, "value");
    if (!lua_isnoneornil(L, 3)) {
        pointer_ref->c_type = luaL_checkstring(L, 3);
    }
    lua_pushvalue(L, 1);
    return 1;
}

static int l_pointer_ref_is_null(lua_State *L) {
    lua_ggml_pointer_ref * pointer_ref = lua_ggml_check_pointer_ref(L, 1);
    lua_pushboolean(L, pointer_ref->ptr == NULL ? 1 : 0);
    return 1;
}

static int l_pointer_ref_clear(lua_State *L) {
    lua_ggml_pointer_ref * pointer_ref = lua_ggml_check_pointer_ref(L, 1);
    pointer_ref->ptr = NULL;
    lua_pushvalue(L, 1);
    return 1;
}

static int l_pointer_ref_type_name(lua_State *L) {
    lua_ggml_pointer_ref * pointer_ref = lua_ggml_check_pointer_ref(L, 1);
    lua_pushstring(L, pointer_ref->c_type != NULL ? pointer_ref->c_type : "void *");
    return 1;
}

static int l_pointer_ref_tostring(lua_State *L) {
    lua_ggml_pointer_ref * pointer_ref = lua_ggml_check_pointer_ref(L, 1);
    lua_pushfstring(L, "pointer_ref<%s>: %p", pointer_ref->c_type != NULL ? pointer_ref->c_type : "void *", pointer_ref->ptr);
    return 1;
}

static int l_pointer_ref(lua_State *L) {
    void * ptr = NULL;
    const char * c_type = NULL;

    if (!lua_isnoneornil(L, 1)) {
        if (lua_isstring(L, 1)) {
            c_type = lua_tostring(L, 1);
        } else {
            ptr = lua_ggml_to_pointer(L, 1, "initial_ptr");
        }
    }

    if (!lua_isnoneornil(L, 2)) {
        c_type = luaL_checkstring(L, 2);
    }

    lua_ggml_push_pointer_ref(L, ptr, c_type);
    return 1;
}

static int l_cplan_gc(lua_State *L) {
    lua_ggml_cplan_ud * cplan = lua_ggml_check_cplan(L, 1);
    free(cplan->plan.work_data);
    cplan->plan.work_data = NULL;
    cplan->plan.work_size = 0;
    return 0;
}

static int l_cplan_free(lua_State *L) {
    lua_ggml_cplan_ud * cplan = lua_ggml_check_cplan(L, 1);
    bool had_data = cplan->plan.work_data != NULL;
    free(cplan->plan.work_data);
    cplan->plan.work_data = NULL;
    cplan->plan.work_size = 0;
    lua_pushboolean(L, had_data ? 1 : 0);
    return 1;
}

static int l_cplan_work_size(lua_State *L) {
    lua_ggml_cplan_ud * cplan = lua_ggml_check_cplan(L, 1);
    lua_pushinteger(L, (lua_Integer) cplan->plan.work_size);
    return 1;
}

static int l_cplan_n_threads(lua_State *L) {
    lua_ggml_cplan_ud * cplan = lua_ggml_check_cplan(L, 1);
    lua_pushinteger(L, (lua_Integer) cplan->plan.n_threads);
    return 1;
}

static int l_cplan_tostring(lua_State *L) {
    lua_ggml_cplan_ud * cplan = lua_ggml_check_cplan(L, 1);
    lua_pushfstring(L, "ggml_cplan: work_size=%d n_threads=%d", (int) cplan->plan.work_size, cplan->plan.n_threads);
    return 1;
}

void lua_ggml_register_metatables(lua_State *L) {
    static const luaL_Reg handle_meta[] = {
        {"__gc", l_handle_gc},
        {"__tostring", l_handle_tostring},
        {NULL, NULL},
    };
    static const luaL_Reg handle_methods[] = {
        {"free", l_handle_free},
        {"get_data", l_tensor_get_data},
        {"is_null", l_handle_is_null},
        {"is_owned", l_handle_is_owned},
        {"name", l_handle_name},
        {"nbytes", l_handle_nbytes},
        {"nelements", l_handle_nelements},
        {"ptr", l_handle_ptr},
        {"set_data", l_tensor_set_data},
        {"shape", l_handle_shape},
        {"type_name", l_handle_type_name},
        {NULL, NULL},
    };
    static const luaL_Reg cplan_meta[] = {
        {"__gc", l_cplan_gc},
        {"__tostring", l_cplan_tostring},
        {NULL, NULL},
    };
    static const luaL_Reg cplan_methods[] = {
        {"free", l_cplan_free},
        {"n_threads", l_cplan_n_threads},
        {"work_size", l_cplan_work_size},
        {NULL, NULL},
    };
    static const luaL_Reg pointer_ref_meta[] = {
        {"__tostring", l_pointer_ref_tostring},
        {NULL, NULL},
    };
    static const luaL_Reg pointer_ref_methods[] = {
        {"clear", l_pointer_ref_clear},
        {"get", l_pointer_ref_get},
        {"is_null", l_pointer_ref_is_null},
        {"set", l_pointer_ref_set},
        {"type_name", l_pointer_ref_type_name},
        {NULL, NULL},
    };

    if (luaL_newmetatable(L, LUA_GGML_HANDLE_MT)) {
        luaL_setfuncs(L, handle_meta, 0);
        lua_newtable(L);
        luaL_setfuncs(L, handle_methods, 0);
        lua_setfield(L, -2, "__index");
    }
    lua_pop(L, 1);

    if (luaL_newmetatable(L, LUA_GGML_CPLAN_MT)) {
        luaL_setfuncs(L, cplan_meta, 0);
        lua_newtable(L);
        luaL_setfuncs(L, cplan_methods, 0);
        lua_setfield(L, -2, "__index");
    }
    lua_pop(L, 1);

    if (luaL_newmetatable(L, LUA_GGML_POINTER_REF_MT)) {
        luaL_setfuncs(L, pointer_ref_meta, 0);
        lua_newtable(L);
        luaL_setfuncs(L, pointer_ref_methods, 0);
        lua_setfield(L, -2, "__index");
    }
    lua_pop(L, 1);
}

static int l_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    struct ggml_context * ctx;

    if (handle != NULL) {
        if (!lua_ggml_handle_is_context(handle)) {
            return luaL_error(L, "ggml.free() expects a ggml_context handle");
        }
        lua_ggml_release_handle(handle);
        return 0;
    }

    ctx = (struct ggml_context *) lua_ggml_to_pointer(L, 1, "ctx");
    ggml_free(ctx);
    return 0;
}

static int l_backend_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    ggml_backend_t backend = (ggml_backend_t) lua_ggml_to_pointer(L, 1, "backend");

    ggml_backend_free(backend);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_backend_buffer_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t) lua_ggml_to_pointer(L, 1, "buffer");

    ggml_backend_buffer_free(buffer);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_gallocr_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    ggml_gallocr_t allocr = (ggml_gallocr_t) lua_ggml_to_pointer(L, 1, "galloc");

    ggml_gallocr_free(allocr);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_backend_sched_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    ggml_backend_sched_t sched = (ggml_backend_sched_t) lua_ggml_to_pointer(L, 1, "sched");

    ggml_backend_sched_free(sched);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_threadpool_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    struct ggml_threadpool * threadpool = (struct ggml_threadpool *) lua_ggml_to_pointer(L, 1, "threadpool");

    ggml_threadpool_free(threadpool);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_backend_event_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    ggml_backend_event_t event = (ggml_backend_event_t) lua_ggml_to_pointer(L, 1, "event");

    ggml_backend_event_free(event);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_gguf_free(lua_State *L) {
    lua_ggml_handle * handle = lua_ggml_test_handle(L, 1);
    struct gguf_context * ctx = (struct gguf_context *) lua_ggml_to_pointer(L, 1, "gguf_ctx");

    gguf_free(ctx);
    if (handle != NULL) {
        lua_ggml_invalidate_handle(L, 1);
    }
    return 0;
}

static int l_graph_plan(lua_State *L) {
    const struct ggml_cgraph * cgraph = (const struct ggml_cgraph *) lua_ggml_to_pointer(L, 1, "cgraph");
    int n_threads = (int) luaL_optinteger(L, 2, GGML_DEFAULT_N_THREADS);
    struct ggml_threadpool * threadpool = (struct ggml_threadpool *) lua_ggml_to_pointer(L, 3, "threadpool");
    struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads, threadpool);
    lua_ggml_cplan_ud * ud = (lua_ggml_cplan_ud *) lua_newuserdata(L, sizeof(*ud));

    ud->plan = plan;
    if (ud->plan.work_size > 0) {
        ud->plan.work_data = (uint8_t *) malloc(ud->plan.work_size);
        if (ud->plan.work_data == NULL) {
            return luaL_error(L, "failed to allocate ggml_cplan work_data (%zu bytes)", ud->plan.work_size);
        }
    }

    luaL_getmetatable(L, LUA_GGML_CPLAN_MT);
    lua_setmetatable(L, -2);
    return 1;
}

static int l_graph_compute(lua_State *L) {
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *) lua_ggml_to_pointer(L, 1, "cgraph");
    lua_ggml_cplan_ud * cplan = lua_ggml_check_cplan(L, 2);
    enum ggml_status status = ggml_graph_compute(cgraph, &cplan->plan);
    lua_pushinteger(L, (lua_Integer) status);
    return 1;
}

static int l_graph_nodes(lua_State *L) {
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *) lua_ggml_to_pointer(L, 1, "cgraph");
    int n_nodes = ggml_graph_n_nodes(cgraph);
    struct ggml_tensor ** nodes = ggml_graph_nodes(cgraph);
    return lua_ggml_push_tensor_table(L, nodes, n_nodes);
}

static int l_build_backward_expand(lua_State *L) {
    struct ggml_context * ctx = (struct ggml_context *) lua_ggml_to_pointer(L, 1, "ctx");
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *) lua_ggml_to_pointer(L, 2, "cgraph");
    size_t count = 0;
    struct ggml_tensor ** grad_accs = NULL;
    int n_nodes = ggml_graph_n_nodes(cgraph);

    if (!lua_isnil(L, 3)) {
        grad_accs = (struct ggml_tensor **) lua_ggml_check_pointer_table(L, 3, "grad_accs", &count, true);
        if ((int) count != n_nodes) {
            free(grad_accs);
            return luaL_error(L, "grad_accs length must match graph node count (%d)", n_nodes);
        }
    }

    ggml_build_backward_expand(ctx, cgraph, grad_accs);
    free(grad_accs);
    return 0;
}

static int l_backend_sched_reserve_size(lua_State *L) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t) lua_ggml_to_pointer(L, 1, "sched");
    struct ggml_cgraph * graph = (struct ggml_cgraph *) lua_ggml_to_pointer(L, 2, "measure_graph");
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    size_t * sizes = (size_t *) calloc((size_t) n_backends, sizeof(*sizes));
    int i;

    if (sizes == NULL) {
        return luaL_error(L, "failed to allocate backend size array");
    }

    ggml_backend_sched_reserve_size(sched, graph, sizes);

    lua_createtable(L, n_backends, 0);
    for (i = 0; i < n_backends; ++i) {
        lua_pushinteger(L, (lua_Integer) sizes[i]);
        lua_rawseti(L, -2, i + 1);
    }

    free(sizes);
    return 1;
}

static int l_backend_compare_graph_backend(lua_State *L) {
    ggml_backend_t backend1 = (ggml_backend_t) lua_ggml_to_pointer(L, 1, "backend1");
    ggml_backend_t backend2 = (ggml_backend_t) lua_ggml_to_pointer(L, 2, "backend2");
    struct ggml_cgraph * graph = (struct ggml_cgraph *) lua_ggml_to_pointer(L, 3, "graph");
    size_t num_test_nodes = 0;
    const struct ggml_tensor * const * test_nodes = NULL;
    bool result;

    if (lua_isfunction(L, 4)) {
        return luaL_error(L, "Lua eval callbacks are not supported safely across ggml worker threads");
    }

    if (!lua_isnoneornil(L, 4)) {
        test_nodes = (const struct ggml_tensor * const *) lua_ggml_check_pointer_table(L, 4, "test_nodes", &num_test_nodes, false);
    }

    result = ggml_backend_compare_graph_backend(backend1, backend2, graph, NULL, NULL, test_nodes, num_test_nodes);
    free((void *) test_nodes);

    lua_pushboolean(L, result ? 1 : 0);
    return 1;
}

static int l_fp32_to_bf16(lua_State *L) {
    ggml_bf16_t value = ggml_fp32_to_bf16((float) luaL_checknumber(L, 1));
    lua_pushinteger(L, (lua_Integer) value.bits);
    return 1;
}

static int l_bf16_to_fp32(lua_State *L) {
    ggml_bf16_t value;
    value.bits = (uint16_t) luaL_checkinteger(L, 1);
    lua_pushnumber(L, (lua_Number) ggml_bf16_to_fp32(value));
    return 1;
}

static int l_format_name(lua_State *L) {
    struct ggml_tensor * tensor = lua_ggml_check_tensor_mut(L, 1, "tensor");
    int top = lua_gettop(L);
    int i;
    const char * name;

    lua_getglobal(L, "string");
    lua_getfield(L, -1, "format");
    lua_remove(L, -2);

    for (i = 2; i <= top; ++i) {
        lua_pushvalue(L, i);
    }

    lua_call(L, top - 1, 1);
    name = luaL_checkstring(L, -1);
    ggml_set_name(tensor, name);
    lua_pop(L, 1);
    lua_pushvalue(L, 1);
    return 1;
}

static int l_abort(lua_State *L) {
    const char * text = luaL_checkstring(L, 1);
    ggml_abort(__FILE__, 0, "%s", text);
    return 0;
}

static int l_log_get(lua_State *L) {
    ggml_log_callback callback = NULL;
    void * user_data = NULL;

    ggml_log_get(&callback, &user_data);

    lua_pushnil(L);
    if (user_data == NULL) {
        lua_pushnil(L);
    } else {
        lua_pushlightuserdata(L, user_data);
    }

    return 2;
}

static int l_log_set(lua_State *L) {
    if (!lua_isnoneornil(L, 1)) {
        return luaL_error(L, "Lua log callbacks are not supported safely across ggml worker threads; pass nil to restore default logging");
    }
    ggml_log_set(NULL, NULL);
    return 0;
}

static int l_set_abort_callback(lua_State *L) {
    if (!lua_isnoneornil(L, 1)) {
        return luaL_error(L, "Lua abort callbacks are not supported safely across ggml worker threads; pass nil to clear");
    }
    ggml_set_abort_callback(NULL);
    lua_pushnil(L);
    return 1;
}

static int l_backend_cpu_set_abort_callback(lua_State *L) {
    ggml_backend_t backend = (ggml_backend_t) lua_ggml_to_pointer(L, 1, "backend_cpu");
    if (!lua_isnoneornil(L, 2)) {
        return luaL_error(L, "Lua abort callbacks are not supported safely across ggml worker threads; pass nil to clear");
    }
    ggml_backend_cpu_set_abort_callback(backend, NULL, NULL);
    return 0;
}

static int l_backend_sched_set_eval_callback(lua_State *L) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t) lua_ggml_to_pointer(L, 1, "sched");
    if (!lua_isnoneornil(L, 2)) {
        return luaL_error(L, "Lua eval callbacks are not supported safely across ggml worker threads; pass nil to clear");
    }
    ggml_backend_sched_set_eval_callback(sched, NULL, NULL);
    return 0;
}

static int l_custom_ops_unsupported(lua_State *L) {
    (void) L;
    return luaL_error(L, "Lua custom op callbacks are not supported safely across ggml worker threads");
}

static const luaL_Reg ggml_funcs[] = {
    {"abort", l_abort},
    {"backend_buffer_free", l_backend_buffer_free},
    {"backend_compare_graph_backend", l_backend_compare_graph_backend},
    {"backend_cpu_set_abort_callback", l_backend_cpu_set_abort_callback},
    {"backend_event_free", l_backend_event_free},
    {"backend_free", l_backend_free},
    {"backend_sched_free", l_backend_sched_free},
    {"backend_sched_reserve_size", l_backend_sched_reserve_size},
    {"backend_sched_set_eval_callback", l_backend_sched_set_eval_callback},
    {"bf16_to_fp32", l_bf16_to_fp32},
    {"build_backward_expand", l_build_backward_expand},
    {"custom_4d", l_custom_ops_unsupported},
    {"custom_inplace", l_custom_ops_unsupported},
    {"format_name", l_format_name},
    {"fp32_to_bf16", l_fp32_to_bf16},
    {"free", l_free},
    {"gallocr_free", l_gallocr_free},
    {"gguf_free", l_gguf_free},
    {"graph_compute", l_graph_compute},
    {"graph_nodes", l_graph_nodes},
    {"graph_plan", l_graph_plan},
    {"log_get", l_log_get},
    {"log_set", l_log_set},
    {"map_custom1", l_custom_ops_unsupported},
    {"map_custom1_inplace", l_custom_ops_unsupported},
    {"map_custom2", l_custom_ops_unsupported},
    {"map_custom2_inplace", l_custom_ops_unsupported},
    {"map_custom3", l_custom_ops_unsupported},
    {"map_custom3_inplace", l_custom_ops_unsupported},
    {"pointer_ref", l_pointer_ref},
    {"set_abort_callback", l_set_abort_callback},
    {"tensor_get_data", l_tensor_get_data},
    {"tensor_set_data", l_tensor_set_data},
    {"threadpool_free", l_threadpool_free},
    {NULL, NULL},
};

int luaopen_ggml(lua_State *L) {
    lua_ggml_register_metatables(L);

    lua_newtable(L);

#if defined(LUA_GGML_HAS_GENERATED_BINDINGS) && LUA_GGML_HAS_GENERATED_BINDINGS
    lua_ggml_add_generated(L);
#endif

    luaL_setfuncs(L, ggml_funcs, 0);

    return 1;
}
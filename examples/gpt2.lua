package.cpath = "./build/?.so;../build/?.so;" .. package.cpath

local ok, ggml = pcall(require, "ggml")
if not ok then
    io.stderr:write("Failed to require 'ggml': " .. tostring(ggml) .. "\n")
    io.stderr:write("Tip: build first with cmake, then run this script from repo root.\n")
    os.exit(1)
end

local GPT2_MAX_NODES = 4096
local EOS_TOKEN_ID = 50256

local function die(msg)
    error(msg, 0)
end

local function usage()
    io.write([[
Usage: lua examples/gpt2.lua [options]

Options:
    -m, --model PATH            Path to GPT-2 GGUF model (default: models/gpt-2-117M/gpt2.gguf)
  -p, --prompt TEXT           Prompt text (approximate greedy tokenizer)
      --prompt-tokens IDS     Comma-separated token ids (recommended for exact behavior)
      --backend NAME          Backend selection: cpu or best (default: cpu)
  -n, --n-predict N           Number of tokens to generate (default: 128)
      --n-ctx N               Runtime context length (default: 1024)
      --n-batch N             Prompt batch size (default: 8)
      --n-threads N           CPU threads (default: 4)
      --top-k N               Top-k sampling (default: 40)
      --top-p P               Top-p sampling (default: 0.95)
      --temp T                Temperature (default: 0.80, <=0 means argmax)
      --seed N                RNG seed (default: current time)
      --ignore-eos            Do not stop at EOS token
  -h, --help                  Show this help

Notes:
    - This script expects a GPT-2 GGUF model file.
  - Text tokenization is an approximate greedy fallback. Use --prompt-tokens for exact input ids.
        - Use --backend best to let ggml pick the best available backend with CPU scheduler fallback.
]])
end

local function parse_int(name, value)
    local n = tonumber(value)
    if not n then
        die("invalid number for " .. name .. ": " .. tostring(value))
    end
    return math.tointeger(n) or n
end

local function parse_float(name, value)
    local n = tonumber(value)
    if not n then
        die("invalid number for " .. name .. ": " .. tostring(value))
    end
    return n
end

local function parse_args(argv)
    local params = {
        model = "",
        prompt = "",
        prompt_tokens = nil,
        backend = "cpu",
        n_predict = 128,
        n_ctx = 1024,
        n_batch = 8,
        n_threads = 4,
        top_k = 40,
        top_p = 0.95,
        temp = 0.80,
        seed = os.time(),
        ignore_eos = false,
    }

    local i = 1
    while i <= #argv do
        local a = argv[i]

        if a == "-h" or a == "--help" then
            usage()
            os.exit(0)
        elseif a == "-m" or a == "--model" then
            i = i + 1
            params.model = argv[i] or die("missing value for " .. a)
        elseif a == "-p" or a == "--prompt" then
            i = i + 1
            params.prompt = argv[i] or die("missing value for " .. a)
        elseif a == "--prompt-tokens" then
            i = i + 1
            params.prompt_tokens = argv[i] or die("missing value for " .. a)
        elseif a == "--backend" then
            i = i + 1
            params.backend = (argv[i] or die("missing value for " .. a)):lower()
        elseif a == "-n" or a == "--n-predict" then
            i = i + 1
            params.n_predict = parse_int(a, argv[i])
        elseif a == "--n-ctx" then
            i = i + 1
            params.n_ctx = parse_int(a, argv[i])
        elseif a == "--n-batch" then
            i = i + 1
            params.n_batch = parse_int(a, argv[i])
        elseif a == "--n-threads" then
            i = i + 1
            params.n_threads = parse_int(a, argv[i])
        elseif a == "--top-k" then
            i = i + 1
            params.top_k = parse_int(a, argv[i])
        elseif a == "--top-p" then
            i = i + 1
            params.top_p = parse_float(a, argv[i])
        elseif a == "--temp" then
            i = i + 1
            params.temp = parse_float(a, argv[i])
        elseif a == "--seed" then
            i = i + 1
            params.seed = parse_int(a, argv[i])
        elseif a == "--ignore-eos" then
            params.ignore_eos = true
        else
            die("unknown argument: " .. tostring(a) .. " (use --help)")
        end

        i = i + 1
    end

    if params.n_ctx <= 0 then die("--n-ctx must be > 0") end
    if params.n_batch <= 0 then die("--n-batch must be > 0") end
    if params.n_threads <= 0 then die("--n-threads must be > 0") end
    if params.n_predict < 0 then die("--n-predict must be >= 0") end
    if params.top_k < 0 then die("--top-k must be >= 0") end
    if params.top_p <= 0.0 or params.top_p > 1.0 then die("--top-p must be in (0, 1]") end
    if params.backend ~= "cpu" and params.backend ~= "best" then
        die("--backend must be one of: cpu, best")
    end

    return params
end

local function build_gpt2_byte_maps()
    local bs = {}
    local cs = {}
    local present = {}
    local n = 0

    local function add_range(first, last)
        for b = first, last do
            bs[#bs + 1] = b
            cs[#cs + 1] = b
            present[b] = true
        end
    end

    add_range(33, 126)
    add_range(161, 172)
    add_range(174, 255)

    for b = 0, 255 do
        if not present[b] then
            bs[#bs + 1] = b
            cs[#cs + 1] = 256 + n
            n = n + 1
        end
    end

    local byte_encoder = {}
    local byte_decoder = {}
    for i = 1, #bs do
        local ch = utf8.char(cs[i])
        byte_encoder[bs[i]] = ch
        byte_decoder[ch] = bs[i]
    end

    return byte_encoder, byte_decoder
end

local function gpt2_decode_token_piece(piece, byte_decoder)
    local out = {}
    local idx = 1

    for _, codepoint in utf8.codes(piece) do
        local ch = utf8.char(codepoint)
        local byte = byte_decoder[ch]
        if byte ~= nil then
            out[idx] = string.char(byte)
        else
            out[idx] = ch
        end
        idx = idx + 1
    end

    return table.concat(out)
end

local function token_to_text(vocab, token_id)
    local piece = vocab.id_to_token[token_id]
    if not piece then
        return nil
    end

    if vocab.byte_decoder then
        return gpt2_decode_token_piece(piece, vocab.byte_decoder)
    end

    return piece
end

local function parse_prompt_tokens(raw)
    local out = {}
    for piece in raw:gmatch("[^,%s]+") do
        local id = tonumber(piece)
        if not id then
            die("invalid token id in --prompt-tokens: " .. tostring(piece))
        end
        out[#out + 1] = math.tointeger(id) or id
    end
    if #out == 0 then
        die("--prompt-tokens must contain at least one token id")
    end
    return out
end

local function tokenize_prompt_greedy(vocab, text)
    local tokens = {}
    local i = 1

    while i <= #text do
        local max_len = math.min(vocab.max_token_len, #text - i + 1)
        local best_id = nil
        local best_len = 0

        for len = max_len, 1, -1 do
            local piece = text:sub(i, i + len - 1)
            local id = vocab.token_to_id[piece]
            if id ~= nil then
                best_id = id
                best_len = len
                break
            end
        end

        if best_id == nil then
            local near = text:sub(i, math.min(#text, i + 20))
            return nil, string.format("cannot tokenize near byte %d (near %q); use --prompt-tokens for exact input", i, near)
        end

        tokens[#tokens + 1] = best_id
        i = i + best_len
    end

    return tokens
end

local function build_kv_cache(model)
    local hp = model.hparams

    model.ctx_kv = ggml.init({
        mem_size = ggml.tensor_overhead() * 2,
        mem_buffer = nil,
        no_alloc = true,
    })
    if not model.ctx_kv then
        die("ggml.init() failed for KV context")
    end

    local n_mem = hp.n_layer * hp.n_ctx
    local n_elements = hp.n_embd * n_mem

    model.memory_k = ggml.new_tensor_1d(model.ctx_kv, ggml.TYPE_F32, n_elements)
    model.memory_v = ggml.new_tensor_1d(model.ctx_kv, ggml.TYPE_F32, n_elements)

    model.buffer_kv = ggml.backend_alloc_ctx_tensors(model.ctx_kv, model.backend)
    if not model.buffer_kv then
        die("ggml.backend_alloc_ctx_tensors() failed for KV cache")
    end

    local mem_size = ggml.backend_buffer_get_size(model.buffer_kv)
    print(string.format("gpt2_model_load: memory size = %8.2f MB, n_mem = %d", mem_size / 1024.0 / 1024.0, n_mem))
end

local function load_vocab_tokens(vocab, tokens)
    if type(tokens) ~= "table" or #tokens == 0 then
        die("GGUF tokenizer.ggml.tokens must be a non-empty string array")
    end

    for i = 1, #tokens do
        local token = tokens[i]
        if type(token) ~= "string" then
            die("GGUF tokenizer.ggml.tokens contains a non-string entry at index " .. i)
        end

        vocab.token_to_id[token] = i - 1
        vocab.id_to_token[i - 1] = token
        if #token > vocab.max_token_len then
            vocab.max_token_len = #token
        end
    end
end

local function gguf_find_required_key_id(gguf_ctx, key)
    local key_id = ggml.gguf_find_key(gguf_ctx, key)
    if key_id < 0 then
        die("GGUF missing required metadata key: " .. key)
    end
    return key_id
end

local function gguf_get_scalar_by_key_id(gguf_ctx, key_id)
    local kv_type = ggml.gguf_get_kv_type(gguf_ctx, key_id)

    if kv_type == ggml.GGUF_TYPE_UINT8 then
        return ggml.gguf_get_val_u8(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_INT8 then
        return ggml.gguf_get_val_i8(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_UINT16 then
        return ggml.gguf_get_val_u16(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_INT16 then
        return ggml.gguf_get_val_i16(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_UINT32 then
        return ggml.gguf_get_val_u32(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_INT32 then
        return ggml.gguf_get_val_i32(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_FLOAT32 then
        return ggml.gguf_get_val_f32(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_UINT64 then
        return ggml.gguf_get_val_u64(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_INT64 then
        return ggml.gguf_get_val_i64(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_FLOAT64 then
        return ggml.gguf_get_val_f64(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_BOOL then
        return ggml.gguf_get_val_bool(gguf_ctx, key_id)
    end
    if kv_type == ggml.GGUF_TYPE_STRING then
        return ggml.gguf_get_val_str(gguf_ctx, key_id)
    end

    die("unsupported GGUF scalar type for key id " .. tostring(key_id) .. ": " .. tostring(kv_type))
end

local function gguf_get_optional_scalar(gguf_ctx, key)
    local key_id = ggml.gguf_find_key(gguf_ctx, key)
    if key_id < 0 then
        return nil
    end
    return gguf_get_scalar_by_key_id(gguf_ctx, key_id)
end

local function gguf_get_required_scalar(gguf_ctx, key)
    return gguf_get_scalar_by_key_id(gguf_ctx, gguf_find_required_key_id(gguf_ctx, key))
end

local function gguf_get_required_string_array(gguf_ctx, key)
    local key_id = gguf_find_required_key_id(gguf_ctx, key)
    local kv_type = ggml.gguf_get_kv_type(gguf_ctx, key_id)
    local arr_type
    local n
    local out = {}

    if kv_type ~= ggml.GGUF_TYPE_ARRAY then
        die("GGUF key is not an array: " .. key)
    end

    arr_type = ggml.gguf_get_arr_type(gguf_ctx, key_id)
    if arr_type ~= ggml.GGUF_TYPE_STRING then
        die("GGUF key is not a string array: " .. key)
    end

    n = ggml.gguf_get_arr_n(gguf_ctx, key_id)
    for i = 0, n - 1 do
        out[#out + 1] = ggml.gguf_get_arr_str(gguf_ctx, key_id, i)
    end

    return out
end

local function gguf_require_tensor(ctx, ...)
    local names = { ... }

    for i = 1, #names do
        local tensor = ggml.get_tensor(ctx, names[i])
        if tensor then
            return tensor
        end
    end

    if #names == 1 then
        die("GGUF missing required tensor: " .. names[1])
    end

    die("GGUF missing required tensor (tried: " .. table.concat(names, ", ") .. ")")
end

local function init_backends(mode, n_threads)
    local backend
    local cpu_backend = nil
    local backends
    local sched

    ggml.backend_load_all()

    if mode == "cpu" then
        backend = ggml.backend_cpu_init()
        if not backend then
            die("ggml.backend_cpu_init() failed")
        end
        ggml.backend_cpu_set_n_threads(backend, n_threads)
        backends = { backend }
    elseif mode == "best" then
        backend = ggml.backend_init_best()
        if not backend then
            die("ggml.backend_init_best() failed")
        end

        if ggml.backend_is_cpu(backend) then
            ggml.backend_cpu_set_n_threads(backend, n_threads)
            backends = { backend }
        else
            cpu_backend = ggml.backend_cpu_init()
            if not cpu_backend then
                ggml.backend_free(backend)
                die("ggml.backend_cpu_init() failed for scheduler fallback")
            end
            ggml.backend_cpu_set_n_threads(cpu_backend, n_threads)
            backends = { backend, cpu_backend }
        end
    else
        die("unsupported backend mode: " .. tostring(mode))
    end

    sched = ggml.backend_sched_new(backends, nil, #backends, GPT2_MAX_NODES, false, true)
    if not sched then
        if cpu_backend then
            ggml.backend_free(cpu_backend)
        end
        ggml.backend_free(backend)
        die("ggml.backend_sched_new() failed")
    end

    return {
        backend = backend,
        cpu_backend = cpu_backend,
        backends = backends,
        sched = sched,
    }
end

local function build_input_buffer(model)
    local hp = model.hparams

    model.ctx_input = ggml.init({
        mem_size = ggml.tensor_overhead() * 2,
        mem_buffer = nil,
        no_alloc = true,
    })
    if not model.ctx_input then
        die("ggml.init() failed for input context")
    end

    model.embd = ggml.new_tensor_1d(model.ctx_input, ggml.TYPE_I32, hp.n_ctx)
    model.position = ggml.new_tensor_1d(model.ctx_input, ggml.TYPE_I32, hp.n_ctx)
    ggml.set_name(model.embd, "in/embd")
    ggml.set_name(model.position, "in/position")

    model.buffer_input = ggml.backend_alloc_ctx_tensors(model.ctx_input, model.backend)
    if not model.buffer_input then
        die("ggml.backend_alloc_ctx_tensors() failed for input buffer")
    end
end

local function load_model_weights(model, path)
    local file = io.open(path, "rb")
    local data_offset
    local n_tensors
    local total_size = 0

    if not file then
        die("failed to open model file: " .. path)
    end

    model.buffer_w = ggml.backend_alloc_ctx_tensors(model.ctx_w, model.backend)
    if not model.buffer_w then
        file:close()
        die("ggml.backend_alloc_ctx_tensors() failed for model weights")
    end
    ggml.backend_buffer_set_usage(model.buffer_w, ggml.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)

    data_offset = ggml.gguf_get_data_offset(model.gguf_ctx)
    n_tensors = ggml.gguf_get_n_tensors(model.gguf_ctx)

    for tensor_id = 0, n_tensors - 1 do
        local name = ggml.gguf_get_tensor_name(model.gguf_ctx, tensor_id)
        local tensor = ggml.get_tensor(model.ctx_w, name)
        local nbytes
        local file_offset
        local data

        if not tensor then
            file:close()
            die("GGUF tensor not found in weights context: " .. tostring(name))
        end

        nbytes = ggml.nbytes(tensor)
        file_offset = data_offset + ggml.gguf_get_tensor_offset(model.gguf_ctx, tensor_id)
        if not file:seek("set", file_offset) then
            file:close()
            die("failed to seek to tensor data for " .. tostring(name))
        end

        data = file:read(nbytes)
        if not data or #data ~= nbytes then
            file:close()
            die("failed to read tensor data for " .. tostring(name))
        end

        ggml.tensor_set_data(tensor, data)
        total_size = total_size + nbytes
    end

    file:close()
    print(string.format("gpt2_model_load: weights size = %8.2f MB", total_size / 1024.0 / 1024.0))
end

local function gpt2_model_load(path, n_ctx_override, backend_mode, n_threads)
    print(string.format("gpt2_model_load: loading GGUF model from '%s'", path))

    local ctx_ref = ggml.pointer_ref("struct ggml_context *")
    local gguf_ctx = ggml.gguf_init_from_file(path, {
        no_alloc = true,
        ctx = ctx_ref,
    })
    local weights_ctx

    if not gguf_ctx then
        die("gguf_init_from_file() failed for '" .. path .. "'")
    end

    weights_ctx = ctx_ref:get(true)
    if not weights_ctx then
        ggml.gguf_free(gguf_ctx)
        die("gguf_init_from_file() did not produce a weights context")
    end

    local model = {
        hparams = {},
        layers = {},
        tensors = {},
        backend = nil,
        cpu_backend = nil,
        backends = nil,
        sched = nil,
        buffer_w = nil,
        buffer_kv = nil,
        buffer_input = nil,
        ctx_w = weights_ctx,
        ctx_kv = nil,
        ctx_input = nil,
        gguf_ctx = gguf_ctx,
    }
    local vocab = {
        token_to_id = {},
        id_to_token = {},
        max_token_len = 0,
    }
    local hp = model.hparams
    local architecture = gguf_get_optional_scalar(gguf_ctx, "general.architecture")
    local tokenizer_model = gguf_get_optional_scalar(gguf_ctx, "tokenizer.ggml.model")
    local tokens = gguf_get_required_string_array(gguf_ctx, "tokenizer.ggml.tokens")

    if architecture ~= nil and architecture ~= "gpt2" then
        die(string.format("GGUF model architecture '%s' is not supported by this GPT-2 example", tostring(architecture)))
    end

    hp.n_ctx = gguf_get_required_scalar(gguf_ctx, "gpt2.context_length")
    hp.n_embd = gguf_get_required_scalar(gguf_ctx, "gpt2.embedding_length")
    hp.n_layer = gguf_get_required_scalar(gguf_ctx, "gpt2.block_count")
    hp.n_head = gguf_get_required_scalar(gguf_ctx, "gpt2.attention.head_count")
    hp.eps = gguf_get_optional_scalar(gguf_ctx, "gpt2.attention.layer_norm_epsilon") or 1e-5

    load_vocab_tokens(vocab, tokens)
    hp.n_vocab = #tokens
    model.eos_token_id = gguf_get_optional_scalar(gguf_ctx, "tokenizer.ggml.eos_token_id") or EOS_TOKEN_ID

    print(string.format("gpt2_model_load: format  = GGUF"))
    print(string.format("gpt2_model_load: n_vocab = %d", hp.n_vocab))
    print(string.format("gpt2_model_load: n_ctx   = %d", hp.n_ctx))
    print(string.format("gpt2_model_load: n_embd  = %d", hp.n_embd))
    print(string.format("gpt2_model_load: n_head  = %d", hp.n_head))
    print(string.format("gpt2_model_load: n_layer = %d", hp.n_layer))
    if tokenizer_model ~= nil then
        print(string.format("gpt2_model_load: tokenizer = %s", tokenizer_model))
    end

    if tokenizer_model ~= nil and tokenizer_model ~= "gpt2" then
        print(string.format("gpt2_model_load: warning: tokenizer.ggml.model is '%s', continuing with greedy tokenization", tokenizer_model))
    end

    if tokenizer_model == "gpt2" then
        vocab.byte_encoder, vocab.byte_decoder = build_gpt2_byte_maps()
    end

    do
        local backend_state = init_backends(backend_mode, n_threads)
        model.backend = backend_state.backend
        model.cpu_backend = backend_state.cpu_backend
        model.backends = backend_state.backends
        model.sched = backend_state.sched
    end
    print(string.format("gpt2_model_load: backend = %s", ggml.backend_name(model.backend)))

    if n_ctx_override and n_ctx_override > 0 then
        hp.n_ctx = math.min(n_ctx_override, hp.n_ctx)
    end

    model.wte = gguf_require_tensor(weights_ctx, "token_embd.weight")
    model.wpe = gguf_require_tensor(weights_ctx, "position_embd.weight", "pos_embd.weight")
    model.ln_f_g = gguf_require_tensor(weights_ctx, "output_norm.weight")
    model.ln_f_b = gguf_require_tensor(weights_ctx, "output_norm.bias")
    model.lm_head = ggml.get_tensor(weights_ctx, "output.weight") or model.wte

    for i = 0, hp.n_layer - 1 do
        local base = "blk." .. i
        local layer = {
            ln_1_g = gguf_require_tensor(weights_ctx, base .. ".attn_norm.weight"),
            ln_1_b = gguf_require_tensor(weights_ctx, base .. ".attn_norm.bias"),
            ln_2_g = gguf_require_tensor(weights_ctx, base .. ".ffn_norm.weight"),
            ln_2_b = gguf_require_tensor(weights_ctx, base .. ".ffn_norm.bias"),
            c_attn_attn_w = gguf_require_tensor(weights_ctx, base .. ".attn_qkv.weight"),
            c_attn_attn_b = gguf_require_tensor(weights_ctx, base .. ".attn_qkv.bias"),
            c_attn_proj_w = gguf_require_tensor(weights_ctx, base .. ".attn_output.weight"),
            c_attn_proj_b = gguf_require_tensor(weights_ctx, base .. ".attn_output.bias"),
            c_mlp_fc_w = gguf_require_tensor(weights_ctx, base .. ".ffn_up.weight"),
            c_mlp_fc_b = gguf_require_tensor(weights_ctx, base .. ".ffn_up.bias"),
            c_mlp_proj_w = gguf_require_tensor(weights_ctx, base .. ".ffn_down.weight"),
            c_mlp_proj_b = gguf_require_tensor(weights_ctx, base .. ".ffn_down.bias"),
        }

        model.layers[i + 1] = layer
    end

    load_model_weights(model, path)
    build_input_buffer(model)
    build_kv_cache(model)

    return model, vocab
end

local function gpt2_graph(model, n_past, n_tokens)
    local N = n_tokens
    local hp = model.hparams

    local n_embd = hp.n_embd
    local n_layer = hp.n_layer
    local n_ctx = hp.n_ctx
    local n_head = hp.n_head

    local mem_size = ggml.tensor_overhead() * GPT2_MAX_NODES + ggml.graph_overhead_custom(GPT2_MAX_NODES, false)
    local ctx = ggml.init({
        mem_size = mem_size,
        mem_buffer = nil,
        no_alloc = true,
    })
    if not ctx then
        die("ggml.init() failed for graph context")
    end

    local gf = ggml.new_graph_custom(ctx, GPT2_MAX_NODES, false)

    local embd = ggml.view_1d(ctx, model.embd, N, 0)
    ggml.set_name(embd, "embd")

    local position = ggml.view_1d(ctx, model.position, N, 0)
    ggml.set_name(position, "position")

    local inpL = ggml.add(
        ctx,
        ggml.get_rows(ctx, model.wte, embd),
        ggml.get_rows(ctx, model.wpe, position)
    )

    for il = 0, n_layer - 1 do
        local layer = model.layers[il + 1]
        local cur = ggml.norm(ctx, inpL, hp.eps)

        cur = ggml.add(
            ctx,
            ggml.mul(ctx, cur, layer.ln_1_g),
            layer.ln_1_b
        )

        cur = ggml.mul_mat(ctx, layer.c_attn_attn_w, cur)
        cur = ggml.add(ctx, cur, layer.c_attn_attn_b)

        do
            local qkv_nb1 = ggml.row_size(ggml.TYPE_F32, 3 * n_embd)
            local qkv_off = ggml.type_size(ggml.TYPE_F32) * n_embd

            local Qcur = ggml.view_2d(ctx, cur, n_embd, N, qkv_nb1, 0)
            local Kcur = ggml.view_2d(ctx, cur, n_embd, N, qkv_nb1, qkv_off)
            local Vcur = ggml.view_2d(ctx, cur, n_embd, N, qkv_nb1, 2 * qkv_off)

            if N >= 1 then
                local kv_stride = ggml.element_size(model.memory_k) * n_embd
                local kv_offset = kv_stride * (il * n_ctx + n_past)

                local k = ggml.view_1d(ctx, model.memory_k, N * n_embd, kv_offset)
                local v = ggml.view_1d(ctx, model.memory_v, N * n_embd, kv_offset)

                ggml.build_forward_expand(gf, ggml.cpy(ctx, Kcur, k))
                ggml.build_forward_expand(gf, ggml.cpy(ctx, Vcur, v))
            end

            local Q = ggml.permute(
                ctx,
                ggml.cont_3d(ctx, Qcur, n_embd / n_head, n_head, N),
                0, 2, 1, 3
            )

            local K = ggml.permute(
                ctx,
                ggml.reshape_3d(
                    ctx,
                    ggml.view_1d(
                        ctx,
                        model.memory_k,
                        (n_past + N) * n_embd,
                        il * n_ctx * ggml.element_size(model.memory_k) * n_embd
                    ),
                    n_embd / n_head,
                    n_head,
                    n_past + N
                ),
                0, 2, 1, 3
            )

            local KQ = ggml.mul_mat(ctx, K, Q)
            local KQ_scaled = ggml.scale(ctx, KQ, 1.0 / math.sqrt(n_embd / n_head))
            local KQ_masked = ggml.diag_mask_inf(ctx, KQ_scaled, n_past)
            local KQ_soft_max = ggml.soft_max(ctx, KQ_masked)

            local V_trans = ggml.cont_3d(
                ctx,
                ggml.permute(
                    ctx,
                    ggml.reshape_3d(
                        ctx,
                        ggml.view_1d(
                            ctx,
                            model.memory_v,
                            (n_past + N) * n_embd,
                            il * n_ctx * ggml.element_size(model.memory_v) * n_embd
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + N
                    ),
                    1, 2, 0, 3
                ),
                n_past + N,
                n_embd / n_head,
                n_head
            )

            local KQV = ggml.mul_mat(ctx, V_trans, KQ_soft_max)
            local KQV_merged = ggml.permute(ctx, KQV, 0, 2, 1, 3)

            cur = ggml.cont_2d(ctx, KQV_merged, n_embd, N)
        end

        cur = ggml.mul_mat(ctx, layer.c_attn_proj_w, cur)
        cur = ggml.add(ctx, cur, layer.c_attn_proj_b)

        cur = ggml.add(ctx, cur, inpL)
        local inpFF = cur

        cur = ggml.norm(ctx, inpFF, hp.eps)
        cur = ggml.add(ctx, ggml.mul(ctx, cur, layer.ln_2_g), layer.ln_2_b)

        cur = ggml.mul_mat(ctx, layer.c_mlp_fc_w, cur)
        cur = ggml.add(ctx, cur, layer.c_mlp_fc_b)
        cur = ggml.gelu(ctx, cur)

        cur = ggml.mul_mat(ctx, layer.c_mlp_proj_w, cur)
        cur = ggml.add(ctx, cur, layer.c_mlp_proj_b)

        inpL = ggml.add(ctx, cur, inpFF)
    end

    inpL = ggml.norm(ctx, inpL, hp.eps)
    inpL = ggml.add(ctx, ggml.mul(ctx, inpL, model.ln_f_g), model.ln_f_b)

    inpL = ggml.mul_mat(ctx, model.lm_head, inpL)
    ggml.set_name(inpL, "logits")
    ggml.set_output(inpL)

    ggml.build_forward_expand(gf, inpL)

    return gf, ctx
end

local function set_i32_tensor_data(tensor, values)
    local packed = {}

    for i = 1, #values do
        packed[i] = string.pack("<i4", values[i])
    end

    ggml.tensor_set_data(tensor, table.concat(packed))
end

local function gpt2_eval(model, n_threads, n_past, embd_inp)
    local N = #embd_inp
    local gf, ctx = gpt2_graph(model, n_past, N)
    local ok_alloc

    ggml.backend_sched_reset(model.sched)
    ok_alloc = ggml.backend_sched_alloc_graph(model.sched, gf)
    if not ok_alloc then
        ggml.free(ctx)
        die("ggml.backend_sched_alloc_graph() failed")
    end

    local positions = {}
    for i = 1, N do
        positions[i] = n_past + i - 1
    end

    set_i32_tensor_data(model.embd, embd_inp)
    set_i32_tensor_data(model.position, positions)

    if model.cpu_backend then
        ggml.backend_cpu_set_n_threads(model.cpu_backend, n_threads)
    elseif ggml.backend_is_cpu(model.backend) then
        ggml.backend_cpu_set_n_threads(model.backend, n_threads)
    end

    local status = ggml.backend_sched_graph_compute(model.sched, gf)
    if status ~= ggml.STATUS_SUCCESS then
        ggml.free(ctx)
        die("ggml.backend_sched_graph_compute() failed with status " .. tostring(status))
    end

    local logits = ggml.graph_get_tensor(gf, "logits")
    if not logits then
        ggml.free(ctx)
        die("failed to fetch logits tensor")
    end

    local n_vocab = model.hparams.n_vocab
    local logits_bytes = n_vocab * ggml.type_size(ggml.TYPE_F32)
    local offset = (N - 1) * logits_bytes
    local packed = ggml.tensor_get_data(logits, offset, logits_bytes)

    local out = {}
    local p = 1
    for i = 1, n_vocab do
        out[i], p = string.unpack("<f", packed, p)
    end

    ggml.free(ctx)

    return out
end

local function sample_top_k_top_p(logits, top_k, top_p, temp)
    local n = #logits

    if temp <= 0 then
        local best_i = 1
        local best = logits[1]
        for i = 2, n do
            if logits[i] > best then
                best = logits[i]
                best_i = i
            end
        end
        return best_i - 1
    end

    local candidates = {}
    for i = 1, n do
        candidates[i] = {
            id = i - 1,
            logit = logits[i] / temp,
        }
    end

    table.sort(candidates, function(a, b)
        return a.logit > b.logit
    end)

    local k = n
    if top_k and top_k > 0 then
        k = math.min(top_k, n)
    end

    while #candidates > k do
        candidates[#candidates] = nil
    end

    local max_logit = candidates[1].logit
    local sum = 0.0
    for i = 1, #candidates do
        local v = math.exp(candidates[i].logit - max_logit)
        candidates[i].prob = v
        sum = sum + v
    end

    for i = 1, #candidates do
        candidates[i].prob = candidates[i].prob / sum
    end

    if top_p < 1.0 then
        local filtered = {}
        local cumulative = 0.0

        for i = 1, #candidates do
            filtered[#filtered + 1] = candidates[i]
            cumulative = cumulative + candidates[i].prob
            if cumulative >= top_p then
                break
            end
        end

        candidates = filtered

        sum = 0.0
        for i = 1, #candidates do
            sum = sum + candidates[i].prob
        end
        for i = 1, #candidates do
            candidates[i].prob = candidates[i].prob / sum
        end
    end

    local r = math.random()
    local c = 0.0
    for i = 1, #candidates do
        c = c + candidates[i].prob
        if r <= c then
            return candidates[i].id
        end
    end

    return candidates[#candidates].id
end

local function free_model(model)
    if not model then
        return
    end

    if model.sched then
        ggml.backend_sched_free(model.sched)
        model.sched = nil
    end

    if model.buffer_input then
        ggml.backend_buffer_free(model.buffer_input)
        model.buffer_input = nil
    end

    if model.buffer_w then
        ggml.backend_buffer_free(model.buffer_w)
        model.buffer_w = nil
    end

    if model.buffer_kv then
        ggml.backend_buffer_free(model.buffer_kv)
        model.buffer_kv = nil
    end

    if model.ctx_input then
        ggml.free(model.ctx_input)
        model.ctx_input = nil
    end

    if model.ctx_kv then
        ggml.free(model.ctx_kv)
        model.ctx_kv = nil
    end

    if model.ctx_w then
        ggml.free(model.ctx_w)
        model.ctx_w = nil
    end

    if model.gguf_ctx then
        ggml.gguf_free(model.gguf_ctx)
        model.gguf_ctx = nil
    end

    if model.cpu_backend then
        ggml.backend_free(model.cpu_backend)
        model.cpu_backend = nil
    end

    if model.backend then
        ggml.backend_free(model.backend)
        model.backend = nil
    end
end

local function main(argv)
    ggml.time_init()

    local t_main_start_us = ggml.time_us()

    local params = parse_args(argv)

    print(string.format("main: seed = %d", params.seed))
    math.randomseed(params.seed)

    local t_load_start_us = ggml.time_us()
    local model, vocab = gpt2_model_load(params.model, params.n_ctx, params.backend, params.n_threads)
    local t_load_us = ggml.time_us() - t_load_start_us

    do
        local n_tokens = math.min(model.hparams.n_ctx, params.n_batch)
        local n_past = model.hparams.n_ctx - n_tokens
        local gf, gctx = gpt2_graph(model, n_past, n_tokens)
        local reserved = ggml.backend_sched_reserve(model.sched, gf)
        local total_size = 0

        ggml.free(gctx)
        if not reserved then
            free_model(model)
            die("ggml.backend_sched_reserve() failed")
        end

        for i = 1, #model.backends do
            local backend = model.backends[i]
            local mem_size = ggml.backend_sched_get_buffer_size(model.sched, backend)
            total_size = total_size + mem_size
            if #model.backends == 1 then
                io.stderr:write(string.format("main: compute buffer size: %.2f MB\n", mem_size / 1024.0 / 1024.0))
            elseif mem_size > 0 then
                io.stderr:write(string.format("main: compute buffer (%s): %.2f MB\n", ggml.backend_name(backend), mem_size / 1024.0 / 1024.0))
            end
        end

        if #model.backends > 1 then
            io.stderr:write(string.format("main: compute buffer total: %.2f MB\n", total_size / 1024.0 / 1024.0))
        end
    end

    local embd_inp = nil
    if params.prompt_tokens then
        embd_inp = parse_prompt_tokens(params.prompt_tokens)
    else
        if params.prompt == "" then
            params.prompt = "Once upon a time"
        end
        local tok_err = nil
        embd_inp, tok_err = tokenize_prompt_greedy(vocab, params.prompt)
        if not embd_inp then
            free_model(model)
            die("tokenization failed for prompt: " .. tostring(tok_err))
        end
    end

    if #embd_inp >= model.hparams.n_ctx then
        free_model(model)
        die(string.format("prompt is too long: %d tokens >= n_ctx %d", #embd_inp, model.hparams.n_ctx))
    end

    params.n_predict = math.min(params.n_predict, model.hparams.n_ctx - #embd_inp)

    print(string.format("main: prompt: '%s'", params.prompt))
    io.write(string.format("main: number of tokens in prompt = %d, first 8 tokens: ", #embd_inp))
    for i = 1, math.min(8, #embd_inp) do
        io.write(string.format("%d ", embd_inp[i]))
    end
    io.write("\n\n")

    local n_past = 0
    local t_sample_us = 0
    local t_predict_us = 0

    local logits = nil

    do
        local idx = 1
        while idx <= #embd_inp do
            local batch = {}
            while idx <= #embd_inp and #batch < params.n_batch do
                batch[#batch + 1] = embd_inp[idx]
                idx = idx + 1
            end

            local t0 = ggml.time_us()
            logits = gpt2_eval(model, params.n_threads, n_past, batch)
            t_predict_us = t_predict_us + (ggml.time_us() - t0)

            n_past = n_past + #batch

            for i = 1, #batch do
                local tok = token_to_text(vocab, batch[i])
                if tok then
                    io.write(tok)
                end
            end
            io.flush()
        end
    end

    for _ = 1, params.n_predict do
        if n_past >= model.hparams.n_ctx then
            break
        end

        local t_sample_start = ggml.time_us()
        local id = sample_top_k_top_p(logits, params.top_k, params.top_p, params.temp)
        t_sample_us = t_sample_us + (ggml.time_us() - t_sample_start)

        if (not params.ignore_eos) and id == (model.eos_token_id or EOS_TOKEN_ID) then
            break
        end

        local tok = token_to_text(vocab, id)
        if tok then
            io.write(tok)
            io.flush()
        end

        local t0 = ggml.time_us()
        logits = gpt2_eval(model, params.n_threads, n_past, { id })
        t_predict_us = t_predict_us + (ggml.time_us() - t0)

        n_past = n_past + 1
    end

    local t_main_end_us = ggml.time_us()

    io.write("\n\n")
    io.write(string.format("main:     load time = %8.2f ms\n", t_load_us / 1000.0))
    io.write(string.format("main:   sample time = %8.2f ms\n", t_sample_us / 1000.0))
    if n_past > 0 then
        io.write(string.format("main:  predict time = %8.2f ms / %.2f ms per token\n", t_predict_us / 1000.0, (t_predict_us / 1000.0) / n_past))
    else
        io.write(string.format("main:  predict time = %8.2f ms\n", t_predict_us / 1000.0))
    end
    io.write(string.format("main:    total time = %8.2f ms\n", (t_main_end_us - t_main_start_us) / 1000.0))

    free_model(model)
end

local ok_main, err = xpcall(function()
    main(arg)
end, debug.traceback)

if not ok_main then
    io.stderr:write(err .. "\n")
    os.exit(1)
end

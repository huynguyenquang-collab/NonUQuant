from transformers import AutoModelForCausalLM, OPTForCausalLM


def load_model(model, model_type, cache_dir=None):
    if model_type == "opt":
        model = OPTForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", cache_dir=cache_dir, trust_remote_code=True
        )
    return model


def parse_model(model):
    model_text = str(type(model)).lower()
    if "opt" in model_text:
        model_type = "opt"
    elif "mistral" in model_text:
        model_type = "mistral"
    elif "qwen" in model_text:
        model_type = "qwen"
    else:
        # additional rules should be added to support other models
        model_type = "llama"
    print(f"Model type : {model_type}")

    return model_type


def get_module_names(model_type):
    if model_type == "opt":
        return ["q", "k", "v", "o", "up", "down"]
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return ["q", "k", "v", "o", "gate", "up", "down"]


def get_modules(layer, model_type):
    if model_type == "opt":
        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
            layer.fc1,
            layer.fc2,
        ]
    else:
        # llama, mistral, qwen
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]

    return modules


def get_named_modules(layer, model_type):
    return list(zip(get_module_names(model_type), get_modules(layer, model_type)))


def get_sequential(model_type):
    if model_type == "opt":
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ]
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]


def get_named_sequential(layer, model_type):
    return list(zip(get_module_names(model_type), get_sequential(model_type)))


def get_model(model, model_type):
    if model_type == "opt":
        return model.model.decoder
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return model.model


def get_layers(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.layers
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return _model.layers


def get_layers_name(model_type):
    if model_type == "opt":
        return "model.decoder.layers"
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return "model.layers"


def get_embedding(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return [_model.embed_tokens, _model.embed_positions]
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return [_model.embed_tokens]


def get_norm(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.final_layer_norm
    else:
        assert model_type == "llama" or model_type == "mistral" or model_type == "qwen"
        return _model.norm

import torch
# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_projector.builder import build_vision_projector

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
GRAPH_TOKEN_INDEX = -200
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_START_TOKEN = "<GH>"
DEFAULT_GRAPH_END_TOKEN = "</GH>"
DEFAULT_GRAPH_PAD_ID = -500

def disable_torch_init():
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    import tensorlayerx as tlx
    setattr(tlx.nn.Linear, "reset_parameters", lambda self: None)
    setattr(tlx.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def tokenizer_graph_token(prompt, tokenizer, graph_token_index=GRAPH_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_GRAPH_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [graph_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
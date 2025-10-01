import argparse
import torch
import os
from gammagl.utils.conversation import conv_templates, SeparatorStyle
from gammagl.utils.gfm_utils import disable_torch_init, KeywordsStoppingCriteria
from gammagl.utils.gfm_utils import DEFAULT_G_END_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from gammagl.models.graphgpt import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

from torch_geometric.data import Data
import json
import copy
from tqdm import tqdm
import json
import os.path as osp


os.environ['TL_BACKEND'] = 'torch'

def load_graph(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 

def run_eval(args):
    # Load and slice prompting file according to start/end
    prompt_file = load_prompting_file(args.prompting_file)
    args.end_id = min(args.end_id, len(prompt_file))
    prompt_slice = prompt_file[args.start_id:args.end_id]

    # Ensure output directory exists
    os.makedirs(args.output_res_path, exist_ok=True)

    # Directly run evaluation without Ray
    print('--- evaluating without Ray ---')
    ans_jsons = eval_model(args, prompt_slice, args.start_id, args.end_id)
    print('--- done ---')


@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    # Resolve graph tower pretrain path (supports absolute or relative to model dir)
    cfg_pretrain_path = getattr(model.config, "pretrain_graph_model_path", None)
    if args.pretrain_graph_model_path is not None:
        pretrain_path = args.pretrain_graph_model_path
    else:
        if cfg_pretrain_path is None:
            raise ValueError("pretrain_graph_model_path not set; please provide --pretrain_graph_model_path")
        pretrain_path = cfg_pretrain_path if osp.isabs(cfg_pretrain_path) else osp.join(args.model_name, cfg_pretrain_path)
    assert osp.exists(osp.join(pretrain_path, 'config.json')), f'config.json missing at {pretrain_path}'

    clip_graph, args_graph = load_model_pretrained(CLIP, pretrain_path)
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    
    model.get_model().graph_tower = graph_tower.cuda()
    # else:
    #     print('other')
    # print(next(graph_tower.parameters()).dtype)
    graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    res_data = []
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break
        graph_dict = load_graph(instruct_item, args.graph_data_path)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']

        qs = instruct_item["conversations"][0]["value"]
        # if use_graph_start_end:
        #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
        # else:
        #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

        # if "v1" in args.model_name.lower():
        #     conv_mode = "graphchat_v1"
        # else: 
        #     raise ValueError('Don\'t support this model')
        conv_mode = "graphchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        graph_data.graph_node = graph_data.graph_node.to(torch.float16)
        # graph_data.edge_index = graph_data.edge_index.to(torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data.cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)

        res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": outputs}.copy())
        with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
            json.dump(res_data, fout, indent=4)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/home/zbs2/GammaGL_algo/论文复现/GraphLama/GraphGPT_data/GraphGPT")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)
    parser.add_argument("--pretrain_graph_model_path", type=str, default=None,
                        help="Path to graph tower pretrain directory containing config.json (overrides model config)")

    args = parser.parse_args()

    # Run in single process without Ray
    run_eval(args)

# protobuf             4.22.3
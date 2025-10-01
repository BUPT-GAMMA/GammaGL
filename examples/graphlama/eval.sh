export PYTHONPATH=/home/zbs2/GammaGL_algo/论文复现/GraphLama/GammaGL
# Fill in the following path to extract projector for the tuning stage following the instructions.
output_model=/path/to/GraphLama/weights
datapath=path/to/cora_test_instruct_std.json
graph_data_path=path/to/graph_data_all.pt
pretrain_graph_model_path=path/to/graph_cli_gt # remember to download arixv ones
res_path=./output_cora_nc # path to save the results
start_id=0
end_id=20000 # total number of instructions to test
num_gpus=1

export CUDA_VISIBLE_DEVICES=2 # define the gpu id to use

python graphlama_trainer.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --pretrain_graph_model_path ${pretrain_graph_model_path} --end_id ${end_id}

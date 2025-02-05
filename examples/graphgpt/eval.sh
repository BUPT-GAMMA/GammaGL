export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH
# to fill in the following path to extract projector for the second tuning stage!
output_model=/local/yy3/graphgpt/GraphGPT-7B-mix-all # path to the pre-trained model checkpoint
datapath=/local/yy3/graphgpt/data/eval/cora_test_instruct_std.json # path to the instruction datset
graph_data_path=/local/yy3/graphgpt/data/graph_data_all.pt # path to the graph data
res_path=./output_stage_2_cora_nc # path to save the results
start_id=0
end_id=20000 # total number of instructions to test
num_gpus=1

export CUDA_VISIBLE_DEVICES=2 # specify the GPU id

python ./examples/graphgpt/graphgpt_trainer.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
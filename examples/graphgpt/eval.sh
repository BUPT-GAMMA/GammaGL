export PYTHONPATH=/home/zbs2/GammaGL_algo/论文复现/GraphLama/GammaGL
# to fill in the following path to extract projector for the second tuning stage!
output_model=/home/zbs2/GammaGL_algo/论文复现/GraphLama/GraphGPT_data/GraphGPT
datapath=/home/zbs2/GammaGL_algo/论文复现/GraphLama/GraphGPT_data/GraphGPT_eval_dataset/cora_test_instruct_std.json
graph_data_path=/home/zbs2/GammaGL_algo/论文复现/GraphLama/GraphGPT_data/graphdata_all/graph_data_all.pt
pretrain_graph_model_path=/home/zbs2/GammaGL_algo/论文复现/GraphLama/GraphGPT_data/graph_cli_gt
res_path=./output_cora_nc # path to save the results
start_id=0
end_id=20000 # total number of instructions to test
num_gpus=1

export CUDA_VISIBLE_DEVICES=2

python /home/zbs2/GammaGL_algo/论文复现/GraphLama/GammaGL/examples/graphgpt/graphgpt_trainer.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --pretrain_graph_model_path ${pretrain_graph_model_path} --end_id ${end_id}

# TLX 版本运行示例：
# python /home/zbs2/GammaGL_algo/论文复现/GraphLama/GammaGL/examples/graphgpt/graphgpt_trainer_tlx.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --pretrain_graph_model_path ${pretrain_graph_model_path} --end_id ${end_id}

# TL_BACKEND="torch" nohup bash examples/graphgpt/eval.sh > log/test_graphgpt.out &
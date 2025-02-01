#!/bin/bash

export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
model_path="/local/yy3/vicuna-7b-v1.5-16k"
model_base="/local/yy3/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector" #meta-llama/Llama-2-7b-hf
mode="v1" # use 'llaga_llama_2' for llama and "v1" for others
dataset="arxiv" #test dataset
task="nc" #test task
emb="simteg"
use_hop=2
sample_size=10
template="ND" # or ND
output_path="llaga/test.txt"

python llaga/llaga_trainer.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template}
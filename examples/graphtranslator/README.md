# GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks
* Paper link: https://arxiv.org/abs/2402.07197
* Author's code repo: https://github.com/alibaba/GraphTranslator

# How to Run

## 0.Datasets & Models Preparation

Follow the original repo to install all required packages;

Then, Download datasets and model checkpoints used in this project with huggingface.

**ArXiv Dataset**

Download files `bert_node_embeddings.pt`, `graphsage_node_embeddings.pt` and `titleabs.tsv` from [link](https://huggingface.co/datasets/Hualouz/GraphTranslator-arixv) and insert them to `./data`.

```
cd ./data
git lfs install
git clone git@hf.co:datasets/Hualouz/GraphTranslator-arxiv
```

**Translator Model**

Download `bert-base-uncased.zip` from [link](https://huggingface.co/Hualouz/Qformer/tree/main) and unzip it to `./Translator/models`.

```
cd Translator/models
git lfs install
git clone git@hf.co:Hualouz/Qformer
unzip bert-base-uncased.zip
```

**ChatGLM2-6B Model**

Download the `ChatGLM2-6B` model from [link](https://huggingface.co/THUDM/chatglm2-6b) and insert it to `./Translator/models` 

```
cd ./Translator/models
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```

## 1.Pre-training Graph Model Phase
- In the pre-training phase, we employ link prediction as the self-supervised task for pre-training the graph model. Finally, we will get the GraphSAGE node embeddings for each node in the graph.
```
cd ./Producer
TL_BACKEND=torch python Embeddings_GraphSAGE.py
```


## 2.Producer Phase

- Generate node summary text with LLM (ChatGLM2-6B).

```
cd ./Producer
python producer.py
```

## 3.Training Phase

Train the Translator model with the prepared ArXiv dataset.

- Stage 1 Training

Train the Translator for GraphModel-Text alignment. The training configurations are in the file `./Translator/pretrain_arxiv_stage1.yaml`.

```
cd ./Translator
TL_BACKEND=torch python graphtranslator_trainer.py --cfg-path ./pretrain_arxiv_stage1.yaml
```

After stage 1, you will get a model checkpoint stored in `./Translator/model_output/pretrain_arxiv_stage1/checkpoint_0.pth`.

- Stage 2 Training

Train the Translator for GraphModel-LLM alignment. The training configurations are in the file `./Translator/pretrain_arxiv_stage2.yaml`.

```
cd ./Translator
TL_BACKEND=torch python graphtranslator_trainer.py --cfg-path ./pretrain_arxiv_stage2.yaml
```

After stage 2, you will get a model checkpoint stored in `./Translator/model_output/pretrain_arxiv_stage2/checkpoint_0.pth`.

After all the training stages , you will get a model checkpoint that can translate GraphModel information into that the LLM can understand.

## 4.Generate and Evaluate Phase

- generate prediction with the pre-trained Translator model. The generate configurations are in the file `./Translator/pretrain_arxiv_generate_stage2.yaml`. As to the inference efficiency, it may take a while to generate all the predictions and save them into file.

```
cd ./Translator
TL_BACKEND=torch python graphtranslator_eval.py
```

The generated prediction results will be saved in `./data/pred.txt`.

# Dataset Statics
| Dataset | # Nodes | # Edges |  # Classes | 
| :-------: | :-------: | :------: | :------: |
| ogb-arxiv | 169,343 | 1,166,243 | 40 |

# Files Description
- `Producer/Embeddings_GraphSAGE.py`: Pre-training Graph Model Phase, which generates node embeddings using GraphSAGE.
- `Producer/producer.py`: Producer Phase, which generates node summary text with LLM (ChatGLM2-6B).
- `Translator/graphtranslator_trainer.py`: Training Phase, which trains the Translator model for GraphModel-Text and GraphModel-LLM alignment.
- `Translator/graphtranslator_eval.py`: Generate and Evaluate Phase, which generates predictions and evaluates the accuracy of the generated predictions.
- `Translator/arxiv_text_pair_datasets.py`: Dataset preparation for ArXiv text pair dataset.
- `Translator/runner_base.py`: Base class for running the training and evaluation processes.

# Results (dataset: ogb-arxiv)

| Metrics | Paper | Our(torch) |
| :-------: | :-------: | :------: |
| Legality Rate(%) | 97.8 | 98.25 | 
| Top-1 Acc (%) | 28.48 | 27.33 | 
| Top-3 Acc (%) | 37.62 | 38.91 | 
| Top-5 Acc (%) | 39.87 | 40.94 | 
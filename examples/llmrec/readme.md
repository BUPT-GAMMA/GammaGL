# LLMRec: Large Language Models with Graph Augmentation for Recommendation

- Paper: LLMRec: Large Language Models with Graph Augmentation for Recommendation
- Author's code : [https://github.com/HKUDS/LLMRec.git](https://github.com/HKUDS/LLMRec.git)

# Dataset Statics
| Dataset | #U    | #I     | #E | 
| ------- | --------- | ---------- | ---------- | 
| Netflix-Original    | 13187     | 17366     | 68933         
| Netflix-Augmented   | 13187    | 17366    | 26374      |


# How to run
-------
create a fold 'data' and
download dataset from https://drive.google.com/drive/folders/1BGKm3nO4xzhyi_mpKJWcfxgi3sQ2j_Ec?usp=drive_link 

create a fold 'log'






```bash
TL_BACKEND='torch' python ./llmrec.py --dataset netflix
TL_BACKEND='tensorflow' python llmrec.py --dataset netflix

# w/o-u-i
python ./llmrec.py --dataset netflix--aug_sample_rate=0.0

# w/o-u
python ./llmrec.py --dataset netflix--user_cat_rate=0

# w/o-u&i
python ./llmrec.py --dataset netflix--user_cat_rate=0  --item_cat_rate=0

# w/o-prune
python ./llmrec.py --dataset netflix --prune_loss_drop_rate=0

```
Results
-------

| Metrics | Paper      | Our(tensorflow)      | Our(pytorch)      |
| ------- | ---------- | ------------ | ------------ |
| Recall    | 0.0829          | 0.0808 | 0.0715 |
| NDCG   | 0.0347 | 0.0321  | 0.0300 |
| Precision     | 0.0041         | 0.0040 |0.0036 |
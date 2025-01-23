# WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding

- Paper link: [https://pubmed.ncbi.nlm.nih.gov/39130614/]

- Author's code:[https://github.com/Melinda315/WalkLM]

## Dataset

**PubMed**

## How to run

download distilroberta model from huggingface and put it in the `distilroberta-base` folder;

The implementation of embedding generate (`emb.py`)、node classification task(`nc.py`) and link prediction task (`lp.py`);

## Example to run the codes

### step 1: fine-tune language model and generate embeddings

```python
TL_BACKEND='torch' python emb.py
```

### step 2: node classification task

```python
TL_BACKEND='torch' python nc.py
```

### step 3: link prediction task

```python
TL_BACKEND='torch' python lp.py
```

## Results

| Task      | Link Prediction |          | Node Classification |          |
| --------- | --------------- | -------- | ------------------- | -------- |
| Metric    | Macro-F1        | Micro-F1 | Macro-F1            | Micro-F1 |
| **Paper** | 85.65           | 94.16    | 60.42               | 62.33    |
| **Ours**  | 85.67           | 93.90    | 58.88               | 60.34    |

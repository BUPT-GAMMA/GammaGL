from transformers import AutoTokenizer, AutoModel, T5Model, AutoModelWithLMHead
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForPreTraining, BertForPreTraining, BertConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import random
import os
import math
import pickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
import warnings
import pickle as pickle
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import tensorlayerx.optimizers as optim


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tlx.set_device("CPU")
print("TL_BACKEND:", os.environ.get("TL_BACKEND"))


op1 = []
op2 = []
op3 = []
with open("./PubMed/node.dat", "r") as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split("\t")
        op1.append(int(temp1))
        op2.append(temp2)
        op3.append(temp3[:-1])

G = [[] for i in range(len(op3))]

with open("./PubMed/link.dat", "r") as original_meta_file:
    for line in original_meta_file:
        start, end, edge_type, edge_class = line.split("\t")
        G[int(start)].append([int(end), int(edge_type)])

line_idx = op1
rand = random.Random()
patient_patient_path = []
alpha = 0.05
path_length = 1000000
path_num = 450000

dic = {}
for line in range(path_num):
    temp_path = []
    start_path = rand.choice(line_idx)
    temp_path.append([start_path, -1])
    dic[start_path] = 1
    for i in range(path_length):
        cur = temp_path[-1][0]
        if len(G[cur]) > 0:
            if rand.random() >= alpha:
                cur_path = rand.choice(G[cur])
                temp_path.append(cur_path)
                dic[cur_path[0]] = 1
            else:
                break
        else:
            break
    if len(temp_path) >= 2:
        patient_patient_path.append(temp_path)

line_name = {}
line_name[0] = "and"
line_name[1] = "causing"
line_name[2] = "and"
line_name[3] = "in"
line_name[4] = "in"
line_name[5] = "and"
line_name[6] = "in"
line_name[7] = "with"
line_name[8] = "with"
line_name[9] = "and"

with open("./PubMed/output.txt", "w") as f:
    for i in range(len(patient_patient_path)):
        print(
            op2[patient_patient_path[i][0][0]],
            line_name[patient_patient_path[i][1][1]],
            op2[patient_patient_path[i][1][0]],
            end="",
            file=f,
        )
        for j in range(1, len(patient_patient_path[i]) - 2):
            print(
                " " + line_name[patient_patient_path[i][j + 1][1]],
                op2[patient_patient_path[i][j + 1][0]],
                end="",
                file=f,
            )
        if len(patient_patient_path[i]) > 2:
            print(
                " " + line_name[patient_patient_path[i][-1][1]],
                op2[patient_patient_path[i][-1][0]],
                end="",
                file=f,
            )
        print("\n", end="", file=f)


with open("./PubMed/output.txt", "r") as file:
    corpus = [line.rstrip("\n") for line in file.readlines()]
    print(len(corpus))

train_text, val_text = train_test_split(corpus, test_size=0.15, random_state=42)

with open("./PubMed/train_corpus.txt", "w") as file:
    for paragraph in train_text:
        file.write(paragraph + "\n")

with open("./PubMed/val_corpus.txt", "w") as file:
    for paragraph in val_text:
        file.write(paragraph + "\n")


datasets = load_dataset(
    "text",
    data_files={
        "train": "./PubMed/train_corpus.txt",
        "validation": "./PubMed/val_corpus.txt",
    },
)

card = "distilroberta-base"

# tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
# model = AutoModelForPreTraining.from_pretrained(card)

model_path = "./distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForPreTraining.from_pretrained(model_path)


def tokenize_function(samples):
    return tokenizer(samples["text"], truncation=True)


tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="model/test",
    overwrite_output_dir=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
    gradient_accumulation_steps=20,
    num_train_epochs=6,
    learning_rate=0.0005,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
trainer.evaluate()

trainer.save_model("model/xyz")


model_name = "model/xyz"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
op1 = []
op2 = []
op3 = []

entity_count, relation_count = 0, 0

with open("./PubMed/node.dat", "r") as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split("\t")
        op1.append(temp1)
        op2.append(temp2)
        op3.append(temp3)


def get_word_embeddings(word):
    encoded_word = tokenizer.encode(word, add_special_tokens=False)
    tokens_tensor = tlx.convert_to_tensor([encoded_word])
    model.eval()
    output = model(tokens_tensor)
    embeddings = output[0][0].mean(dim=0)
    return tlx.convert_to_numpy(embeddings)


emb = get_word_embeddings("hello")
emb = np.zeros((len(op2), len(emb)))


for i in range(len(op2)):
    word = op2[i]
    emb[i] = get_word_embeddings(word)

with open("./PubMed/emb.dat", "w") as file:
    file.write("pubmed\n")
    for i in range(len(op2)):
        file.write(f"{i}\t")
        file.write(" ".join(emb[i].astype(str)))
        file.write("\n")

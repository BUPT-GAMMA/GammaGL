import openai
from openai import OpenAI
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import time
import re
from datetime import datetime, timedelta, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# model_list = ["text-davinci-003", "code-davinci-002", "gpt-3.5-turbo", "gpt-4"]
model_list = ["qwen-max"]
parser = argparse.ArgumentParser(description="GNN")
"""
parser.add_argument(
    "--model",
    type=str,
    default="text-davinci-003",
    help="name of LM (default: text-davinci-003)",
)
"""
parser.add_argument(
    "--model",
    type=str,
    default="qwen-max",
    help="name of LM (default: qwen-max)",
)
parser.add_argument("--mode", type=str, default="easy", help="mode (default: easy)")
parser.add_argument(
    "--prompt", type=str, default="none", help="prompting techniques (default: none)"
)
parser.add_argument("--T", type=int, default=0, help="temprature (default: 0)")
parser.add_argument("--token", type=int, default=2000, help="max token (default: 400)")
parser.add_argument(
    "--layer", type=int, default=1, help="number of GNN layers (default: 1)"
)
parser.add_argument("--SC", type=int, default=0, help="self-consistency (default: 0)")
parser.add_argument(
    "--SC_num",
    type=int,
    default=5,
    help="number of cases for self-consistency (default: 5)",
)
args = parser.parse_args()
assert args.prompt in ["CoT", "none", "0-CoT", "LTM", "PROGRAM", "k-shot"]
assert args.layer in [1, 2]


def translate(G, embedding, args):
    edge = list(G.edges())
    n, m = G.number_of_nodes(), G.number_of_edges()
    Q = ""
    if args.prompt in ["CoT", "k-shot"]:
        if args.layer == 2:
            with open("NLGraph/GNN/prompt/" + args.prompt + "-prompt.txt", "r") as f:
                exemplar = f.read()
        else:
            with open(
                "NLGraph/GNN/one-prompt/" + args.prompt + "-prompt.txt", "r"
            ) as f:
                exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = (
        Q
        + "In an undirected graph, the nodes are numbered from 0 to "
        + str(n - 1)
        + ", and every node has an embedding. (i,j) means that node i and node j are connected with an undirected edge.\n"
    )
    Q = Q + "Embeddings:\n"
    for i in range(n):
        Q = Q + "node " + str(i) + ": ["
        for j in range(len(embedding[i]) - 1):
            Q = Q + str(embedding[i][j]) + ","
        Q = Q + str(embedding[i][-1]) + "]\n"
    Q = Q + "The edges are:"
    # character = [chr(65+i) for i in range(26)] + [chr(65+i)+chr(65+i) for i in range(26)]
    for i in range(len(edge)):
        Q = Q + " (" + str(edge[i][0]) + "," + str(edge[i][1]) + ")"
    Q = Q + "\n"
    Q = (
        Q
        + "In a simple graph convolution layer, each node's embedding is updated by the sum of its neighbors' embeddings.\n"
    )
    word = "two layers"
    if args.layer == 1:
        word = "one layer"
    Q = (
        Q
        + "Q: What's the embedding of each node after "
        + word
        + " of simple graph convolution layer?"
        + "\nA:"
    )
    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's think step by step:"
        case "LTM":
            Q = Q + " Let's break down this problem:"
        case "PROGRAM":
            Q = Q + " Let's solve the problem by a Python program:"
    return Q


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1000))
def predict(Q, args):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    input = Q
    temperature = 0
    if args.SC == 1:
        temperature = 0.7

    Answer_list = []
    for text in input:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=args.token,
        )
        # Answer_list.append(response["choices"][0]["message"]["content"])
        Answer_list.append(response.choices[0].message.content)
        # print(Answer_list)
    return Answer_list


def log(Q, res, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = "log/GNN/" + args.model + "-" + args.mode + "-" + time + "-" + args.prompt
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath + "res.npy", res)
    np.save(newpath + "answer.npy", answer)
    with open(newpath + "prompt.txt", "w") as f:
        f.write(Q)
        f.write("\n")
        f.write("Acc: " + str(res.sum()) + "/" + str(len(res)) + "\n")
        f.write("\n")
        print(args, file=f)


def evaluate(ans, std, G):
    n = G.number_of_nodes()
    mode_string = "node " + str(0)
    pos = ans.rfind(mode_string)
    if pos == -1:
        return 0
    answer = list(re.findall("\d+", ans[pos:]))
    flag = 1
    for i in range(n):
        for j in range(len(std[i])):
            if abs(float(answer[i * (len(std[i]) + 1) + j + 1]) - std[i][j]) > 0.01:
                return 0
    return flag


def main():
    """
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    """
    res, answer = [], []
    match args.mode:
        case "easy":
            g_num = 100
        case "hard":
            g_num = 200

    batch_num = 10
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, std_list = [], [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            with open(
                "NLgraph/GNN/graph/" + args.mode + "/standard/graph" + str(j) + ".txt",
                "r",
            ) as f:
                n, m, d = [int(x) for x in next(f).split()]
                array = []
                for line in f:  # read rest of lines
                    array.append([int(x) for x in line.split()])
                edge, embedding = array[:m], array[m : m + n]
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(int(edge[k][0]), int(edge[k][1]))
                num = n
                embedding_dim = 2
                std = np.zeros((num, embedding_dim))
                embedding1 = np.zeros((num, embedding_dim))
                for i in range(G.number_of_nodes()):
                    cnt_v = np.zeros(embedding_dim)
                    for jj in G[i]:
                        cnt_v += embedding[jj]
                    embedding1[i] = cnt_v
                for i in range(G.number_of_nodes()):
                    cnt_v = np.zeros(embedding_dim)
                    for jj in G[i]:
                        cnt_v += embedding1[jj]
                    std[i] = cnt_v
                Q = translate(G, embedding, args)
                Q_list.append(Q)
                G_list.append(G)
                if args.layer == 2:
                    std_list.append(std)
                else:
                    std_list.append(embedding1)
        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            sc_list.append(answer_list)
        for j in range(len(Q_list)):
            vote = 0
            for k in range(sc):
                ans, G, std = sc_list[k][j].lower(), G_list[j], std_list[j]
                answer.append(ans.lower())
                # try:

                r = evaluate(ans.lower(), std, G)
                vote += r
                # except:
                # print(ans.lower())
            r = 1 if vote * 2 > sc else 0
            res.append(r)
    res = np.array(res)
    answer = np.array(answer)
    log(Q, res, answer, args)
    print(res.sum())


if __name__ == "__main__":
    main()

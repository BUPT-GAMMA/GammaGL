import openai
from openai import OpenAI
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import time
from datetime import datetime, timedelta, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# model_list = ["text-davinci-003", "code-davinci-002", "gpt-3.5-turbo", "gpt-4"]
model_list = ["qwen-max"]
parser = argparse.ArgumentParser(description="maximum flow")
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
parser.add_argument("--token", type=int, default=400, help="max token (default: 400)")
parser.add_argument("--SC", type=int, default=0, help="self-consistency (default: 0)")
parser.add_argument(
    "--SC_num",
    type=int,
    default=5,
    help="number of cases for self-consistency (default: 5)",
)
args = parser.parse_args()
assert args.prompt in ["CoT", "none", "0-CoT", "LTM", "PROGRAM", "k-shot"]


def translate(G, q, args):
    edge = list(G.edges())
    n, m = G.number_of_nodes(), G.number_of_edges()
    Q = ""
    if args.prompt in ["CoT", "k-shot"]:
        with open("NLGraph/flow/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = (
        Q
        + "In a directed graph, the nodes are numbered from 0 to "
        + str(n - 1)
        + ", and the edges are:\n"
    )
    for i in range(len(edge)):
        Q = (
            Q
            + "an edge from node "
            + str(edge[i][0])
            + " to node "
            + str(edge[i][1])
            + " with capacity "
            + str(G[edge[i][0]][edge[i][1]]["capacity"])
        )
        if i + 1 == len(edge):
            Q = Q + "."
        else:
            Q = Q + ","
        Q = Q + "\n"
    Q = (
        Q
        + "Q: What is the maximum flow from node "
        + str(q[0])
        + " to node "
        + str(q[1])
        + "?"
    )
    Q = Q + "\nA:"
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


def log(Q, res1, res2, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = (
        "log/flow/" + args.model + "-" + args.mode + "-" + time + "-" + args.prompt
    )
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath + "res1.npy", res1)
    np.save(newpath + "res2.npy", res2)
    np.save(newpath + "answer.npy", answer)
    with open(newpath + "prompt.txt", "w") as f:
        f.write(Q)
        f.write("\n")
        f.write("Acc: " + str(res1.sum()) + "/" + str(len(res1)) + "\n")
        f.write("Acc2: " + str(res2.sum()) + "/" + str(len(res2)) + "\n")
        f.write("\n")
        print(args, file=f)


def evaluate(ans, G, q, std):
    mode_str = "the maximum flow from node " + str(q[0]) + " to node " + str(q[1])
    pos = ans.find(mode_str)
    if pos == -1:
        return 0, 0
    flag1, flag2 = 1, 1
    pos = pos + len(mode_str) + 1
    i = pos
    while i < len(ans) and not (ans[i] >= "0" and ans[i] <= "9"):
        i += 1
    num = 0
    while i < len(ans) and ans[i] >= "0" and ans[i] <= "9":
        num = num * 10 + int(ans[i])
        i += 1
    if num != std:
        flag2 = 0

    return flag1, flag2


def main():
    """
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        raise Exception("Missing openai key!")
    if "OPENAI_ORGANIZATION" in os.environ:
        openai.organization = os.environ["OPENAI_ORGANIZATION"]
    """
    res1, res2, answer = [], [], []
    match args.mode:
        case "easy":
            g_num = 150
        case "hard":
            g_num = 200

    batch_num = 20
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, q_list, std_list = [], [], [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            with open(
                "NLgraph/flow/graph/" + args.mode + "/standard/graph" + str(j) + ".txt",
                "r",
            ) as f:
                n, m = [int(x) for x in next(f).split()]
                array = []
                for line in f:  # read rest of lines
                    array.append([int(x) for x in line.split()])
                edge, q, std = array[:-2], array[-2], array[-1][0]
                G = nx.DiGraph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1], capacity=edge[k][2])
                Q = translate(G, q, args)
                Q_list.append(Q)
                G_list.append(G)
                q_list.append(q)
                std_list.append(std)
        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            sc_list.append(answer_list)
        for j in range(len(Q_list)):
            vote1, vote2 = 0, 0
            for k in range(sc):
                ans, G, std = sc_list[k][j].lower(), G_list[j], std_list[j]
                answer.append(ans.lower())
                try:
                    r1, r2 = evaluate(
                        ans.lower(), G, q_list[j], std
                    )  # r1 for solution check and r2 for final answer check
                    vote1 += r1
                    vote2 += r2
                except:
                    print(ans.lower())
            r1 = 1 if vote1 * 2 > sc else 0
            r2 = 1 if vote2 * 2 > sc else 0
            res1.append(r1)
            res2.append(r2)

    res1 = np.array(res1)
    res2 = np.array(res2)
    answer = np.array(answer)
    log(Q, res1, res2, answer, args)
    print(res2.sum())


if __name__ == "__main__":
    main()

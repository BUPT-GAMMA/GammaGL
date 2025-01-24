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
model_list = ["qwen-vl-plus"]
parser = argparse.ArgumentParser(description="shortest path")
parser.add_argument(
    "--model",
    type=str,
    default="qwen-vl-plus",
    help="name of LM (default: qwen-vl-plus)",
)
parser.add_argument("--mode", type=str, default="easy", help="mode (default: easy)")
parser.add_argument(
    "--prompt", type=str, default="CoT", help="prompting techniques (default: none)"
)
parser.add_argument("--T", type=int, default=0, help="temprature (default: 0)")
parser.add_argument("--token", type=int, default=400, help="max token (default: 400)")
parser.add_argument("--SC", type=int, default=1, help="self-consistency (default: 0)")
parser.add_argument(
    "--city", type=int, default=0, help="whether to use city (default: 0)"
)
parser.add_argument(
    "--SC_num",
    type=int,
    default=5,
    help="number of cases for self-consistency (default: 5)",
)
args = parser.parse_args()
assert args.prompt in [
    "CoT",
    "none",
    "0-CoT",
    "LTM",
    "PROGRAM",
    "k-shot",
    "Algorithm",
    "Instruct",
    "dot1",
    "dot2",
    "ins1",
    "ins2",
    "ins3",
]


def translate(G, q, args):
    edge = list(G.edges())
    n, m = G.number_of_nodes(), G.number_of_edges()
    Q = ""
    prompt_folder = "prompt"
    if args.city == 1:
        prompt_folder = "city-prompt"
    if args.prompt in [
        "CoT",
        "k-shot",
        "Algorithm",
        "Instruct",
        "dot1",
        "dot2",
        "ins1",
        "ins2",
        "ins3",
    ]:
        with open(
            "NLGraph/shortest_path/"
            + prompt_folder
            + "/"
            + args.prompt
            + "-prompt.txt",
            "r",
        ) as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    if args.city == 0:
        Q = (
            Q
            + "In an undirected graph, the nodes are numbered from 0 to "
            + str(n - 1)
            + ", and the edges are:\n"
        )
    else:
        Q = (
            Q
            + "In a country, the cities are numbered from 0 to "
            + str(n - 1)
            + ". There are roads between the cities, and the roads are:\n"
        )
    if args.city == 0:
        for i in range(len(edge)):
            Q = (
                Q
                + "an edge between node "
                + str(edge[i][0])
                + " and node "
                + str(edge[i][1])
                + " with weight "
                + str(G[edge[i][0]][edge[i][1]]["weight"])
            )
            if i + 1 == len(edge):
                Q = Q + "."
            else:
                Q = Q + ","
            Q = Q + "\n"
    else:
        for i in range(len(edge)):
            Q = (
                Q
                + "a road between city "
                + str(edge[i][0])
                + " and city "
                + str(edge[i][1])
                + " with length "
                + str(G[edge[i][0]][edge[i][1]]["weight"])
            )
            if i + 1 == len(edge):
                Q = Q + "."
            else:
                Q = Q + ","
            Q = Q + "\n"
    if args.prompt == "Instruct":
        Q = Q + "Let's construct a graph with the nodes and edges first.\n"
    elif args.prompt == "ins1":
        Q = Q + "Let's think step by step.\n"
    elif args.prompt == "ins2":
        Q = Q + "Examine each detail carefully.\n"
    elif args.prompt == "ins3":
        Q = Q + "Think it through systematically.\n"

    elif args.prompt == "dot1":
        for num in range(55):
            Q = Q + "."
        Q = Q + "\n"
    if args.city == 0:
        Q = (
            Q
            + "Q: Give the shortest path from node "
            + str(q[0])
            + " to node "
            + str(q[1])
            + ".\nA:"
        )
    else:
        Q = (
            Q
            + "Q: Give the shortest path from city "
            + str(q[0])
            + " to city "
            + str(q[1])
            + ".\nA:"
        )

    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's think step by step:"
        case "LTM":
            Q = Q + " Let's break down this problem:"
        case "PROGRAM":
            Q = Q + " Let's solve the problem by a Python program:"
    return Q


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(1000))
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
        "log/shortest_path/"
        + args.model
        + "-"
        + args.mode
        + "-"
        + time
        + "-"
        + args.prompt
    )
    if args.city == 1:
        newpath = newpath + "+city"
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


def evaluate(ans, G, q):
    entity = "node"
    if args.city == 1:
        entity = "city"
    mode_str = (
        "the shortest path from "
        + entity
        + " "
        + str(q[0])
        + " to "
        + entity
        + " "
        + str(q[1])
    )
    pos = ans.find(mode_str)
    if pos == -1:
        return 0, 0
    pos = pos + len(mode_str) + 1
    num, flag = 0, 0
    solution = []
    for i in range(pos, len(ans)):
        if ans[i] >= "0" and ans[i] <= "9":
            num = num * 10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if num == q[1]:
                    break
                flag = 0
            num = 0
    length = 0
    flag1, flag2 = 1, 1
    for i in range(len(solution) - 1):
        if not G.has_edge(solution[i], solution[i + 1]):
            flag1 = 0
            break
        length += G[solution[i]][solution[i + 1]]["weight"]
    shortest = nx.shortest_path_length(G, source=q[0], target=q[1], weight="weight")
    if length != shortest:
        flag1 = 0
    pos = ans.rfind("total length")
    if pos == -1:
        return flag1, 0
    i = pos
    while i < len(ans) and not (ans[i] >= "0" and ans[i] <= "9"):
        i += 1
    num = 0
    while i < len(ans) and ans[i] >= "0" and ans[i] <= "9":
        num = num * 10 + int(ans[i])
        i += 1
    if num != shortest:
        flag2 = 0
    return flag1, flag2


def main():
    """
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    """
    if "OPENAI_ORGANIZATION" in os.environ:
        openai.organization = os.environ["OPENAI_ORGANIZATION"]
    res1, res2, answer = [], [], []
    match args.mode:
        case "easy":
            g_num = 20
        case "hard":
            g_num = 200

    batch_num = 4
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, q_list = [], [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            with open(
                "NLgraph/shortest_path/graph/"
                + args.mode
                + "/standard/graph"
                + str(j)
                + ".txt",
                "r",
            ) as f:
                n, m = [int(x) for x in next(f).split()]
                array = []
                for line in f:  # read rest of lines
                    array.append([int(x) for x in line.split()])
                edge, q = array[:-1], array[-1]
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1], weight=edge[k][2])
                Q = translate(G, q, args)
                Q_list.append(Q)
                G_list.append(G)
                q_list.append(q)
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
                ans, G = sc_list[k][j].lower(), G_list[j]
                answer.append(ans.lower())
                try:
                    r1, r2 = evaluate(
                        ans.lower(), G, q_list[j]
                    )  # r1 for path_length check and r2 for total weight check
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

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
parser = argparse.ArgumentParser(description="matching")
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
assert args.prompt in [
    "CoT",
    "none",
    "0-CoT",
    "LTM",
    "PROGRAM",
    "k-shot",
    "Instruct",
    "Algorithm",
]


def translate(G, n1, n2, args):
    edge = list(G.edges())
    m = G.number_of_edges()
    Q = ""
    if args.prompt in ["CoT", "k-shot", "Instruct", "Algorithm"]:
        with open("NLGraph/matching/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = (
        Q
        + "There are "
        + str(n1)
        + " job applicants numbered from 0 to "
        + str(n1 - 1)
        + ", and "
        + str(n2)
        + " jobs numbered from 0 to "
        + str(n2 - 1)
        + ". Each applicant is interested in some of the jobs. Each job can only accept one applicant and a job applicant can be appointed for only one job.\n"
    )

    # character = [chr(65+i) for i in range(26)] + [chr(65+i)+chr(65+i) for i in range(26)]
    for i in range(len(edge)):
        Q = (
            Q
            + "Applicant "
            + str(edge[i][0])
            + " is interested in job "
            + str(edge[i][1] - n1)
            + ".\n"
        )
    Q = (
        Q
        + "Q: Find an assignment of jobs to applicants in such that the maximum number of applicants find the job they are interested in."
    )
    # if args.prompt != "none":
    #     Q = Q + " Give the solution."
    Q = Q + "\n"
    if args.prompt == "Instruct":
        Q = Q + "Let's construct a graph based on the above information first.\n"
    Q = Q + "A:"
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


def log(Q, res1, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = (
        "log/matching/" + args.model + "-" + args.mode + "-" + time + "-" + args.prompt
    )
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath + "res1.npy", res1)
    np.save(newpath + "answer.npy", answer)
    with open(newpath + "prompt.txt", "w") as f:
        f.write(Q)
        f.write("\n")
        f.write("Acc: " + str(res1.sum()) + "/" + str(len(res1)) + "\n")
        f.write("\n")
        print(args, file=f)


def evaluate(ans, G, n1, std):

    i = ans.find("applicant")
    j = ans.find("job")
    tag = 0
    if j < i:
        tag = 1
        i = j
    if i == -1:
        return 0, 0
    solution = []
    pos = ans.find("this way")
    i = max(i, ans.find("make the following assignments"))
    if pos == -1:
        pos = len(ans)
    while i < pos:
        while i < pos and not (ans[i] >= "0" and ans[i] <= "9"):
            i += 1
        if i == pos:
            break
        num = 0
        while i < pos and ans[i] >= "0" and ans[i] <= "9":
            num = num * 10 + int(ans[i])
            i += 1
        while i < pos and not (ans[i] >= "0" and ans[i] <= "9"):
            i += 1
        if i == pos:
            break
        num2 = 0
        while i < pos and ans[i] >= "0" and ans[i] <= "9":
            num2 = num2 * 10 + int(ans[i])
            i += 1
        if tag:
            solution.append((num2, num + n1))
        else:
            solution.append((num, num2 + n1))
    cnt = 0
    for edg in solution:
        if G.has_edge(edg[0], edg[1]):
            cnt += 1
    for i in range(len(solution)):
        for j in range(i - 1):
            if solution[i][0] == solution[j][0] or solution[i][1] == solution[j][1]:
                return 0
    if cnt != std:
        return 0
    return 1


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
            g_num = 300
        case "hard":
            g_num = 210

    batch_num = 20
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, std_list, n1_list = [], [], [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            with open(
                "log/matching/" + args.mode + "/standard/graph" + str(j) + ".txt", "r"
            ) as f:
                n1, n2, m = [int(x) for x in next(f).split()]
                array = []
                for line in f:  # read rest of lines
                    array.append([int(x) for x in line.split()])
                edge, std = array[:-1], array[-1][0]
                G = nx.Graph()
                G.add_nodes_from(range(n1), bipartite=0)
                G.add_nodes_from(range(n1, n1 + n2), bipartite=1)
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q = translate(G, n1, n2, args)
                Q_list.append(Q)
                G_list.append(G)
                n1_list.append(n1)
                std_list.append(std)
        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            sc_list.append(answer_list)
        for j in range(len(Q_list)):
            vote1 = 0
            for k in range(sc):
                ans, G, std, n1 = (
                    sc_list[k][j].lower(),
                    G_list[j],
                    std_list[j],
                    n1_list[j],
                )
                answer.append(ans.lower())
                try:
                    r1 = evaluate(
                        ans.lower(), G, n1, std
                    )  # r1 for solution check and r2 for final answer check
                    vote1 += r1
                except:
                    print(ans.lower())
            r1 = 1 if vote1 * 2 > sc else 0
            res1.append(r1)

    res1 = np.array(res1)
    answer = np.array(answer)
    log(Q, res1, answer, args)
    print(res1.sum())


if __name__ == "__main__":
    main()

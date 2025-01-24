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

# model_list = ["text-davinci-003","code-davinci-002","gpt-3.5-turbo","gpt-4"]
model_list = ["qwen-max"]
parser = argparse.ArgumentParser(description="connectivity")
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
    "--prompt", type=str, default="CoT", help="prompting techniques (default: CoT)"
)
parser.add_argument("--T", type=int, default=0, help="temprature (default: 0)")
parser.add_argument("--token", type=int, default=256, help="max token (default: 256)")
parser.add_argument("--SC", type=int, default=0, help="self-consistency (default: 0)")
parser.add_argument(
    "--SC_num", type=int, default=5, help="number of cases for SC (default: 5)"
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
]


def translate(m, q, array, args):
    edge = array[:m]
    question = array[m:]
    Q = ""
    if args.prompt in ["CoT", "k-shot", "Algorithm", "Instruct"]:
        with open(
            "NLGraph/connectivity/prompt/" + args.prompt + "-prompt.txt", "r"
        ) as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = (
        Q
        + "Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.\nGraph:"
    )
    for i in range(m):
        Q = Q + " (" + str(edge[i][0]) + "," + str(edge[i][1]) + ")"
    Q = Q + "\n"

    if args.prompt == "Instruct":
        Q = Q + "Let's construct a graph with the nodes and edges first.\n"
    Q = Q + "Q: Is there a path between "
    Q_list = []
    for i in range(q * 2):
        Q_i = (
            Q
            + "node "
            + str(question[i][0])
            + " and node "
            + str(question[i][1])
            + "?\nA:"
        )
        match args.prompt:
            case "0-CoT":
                Q_i = Q_i + " Let's think step by step:"
            case "LTM":
                Q_i = Q_i + " Let's break down this problem:"
            case "PROGRAM":
                Q_i = Q_i + " Let's solve the problem by a Python program:"
        Q_list.append(Q_i)
    return Q_list


"""
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1000))

def predict(Q, args):
    input = Q
    temperature = args.T
    if args.SC == 1:
        temperature = 0.7
    if "gpt" in args.model:
        Answer_list = []
        for text in input:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text},
                ],
                temperature=temperature,
                max_tokens=args.token,
            )
            Answer_list.append(response["choices"][0]["message"]["content"])
        return Answer_list
    response = openai.Completion.create(
        model=args.model,
        prompt=input,
        temperature=temperature,
        max_tokens=args.token,
    )
    Answer_list = []
    for i in range(len(input)):
        Answer_list.append(response["choices"][i]["text"])
    return Answer_list
"""


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1000))
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
        Answer_list.append(response.choices[0].message.content)
    return Answer_list

    """
    response = openai.Completion.create(
    model=args.model,
    prompt=input,
    temperature=temperature,
    max_tokens=args.token,
    )
    Answer_list = []
    for i in range(len(input)):
        Answer_list.append(response["choices"][i]["text"])
    return Answer_list
    """


def log(Q_list, res, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = (
        "log/connectivity/"
        + args.model
        + "-"
        + args.mode
        + "-"
        + time
        + "-"
        + args.prompt
    )
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath + "res.npy", res)
    np.save(newpath + "answer.npy", answer)
    with open(newpath + "prompt.txt", "w") as f:
        f.write(Q_list[0])
        f.write("\n")
        f.write("Acc: " + str(res.sum()) + "/" + str(len(res)) + "\n")
        print(args, file=f)


def main():
    """
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_KEY"]
    else:
        raise Exception("Missing openai key!")
    if "OPENAI_ORGANIZATION" in os.environ:
        openai.organization = os.environ["OPENAI_ORGANIZATION"]
    """
    res, answer = [], []
    match args.mode:
        case "easy":
            # g_num = 36
            g_num = 8
        case "medium":
            g_num = 120
        case "hard":
            g_num = 68
    for i in tqdm(range(g_num)):
        with open(
            "NLgraph/connectivity/graph/"
            + args.mode
            + "/standard/graph"
            + str(i)
            + ".txt",
            "r",
        ) as f:
            n, m, q = [int(x) for x in next(f).split()]
            array = []
            for line in f:  # read rest of lines
                array.append([int(x) for x in line.split()])
            qt = array[m:]
            Q_list = translate(m, q, array, args)
            sc = 1
            if args.SC == 1:
                sc = args.SC_num
            sc_list = []
            for k in range(sc):
                answer_list = predict(Q_list, args)
                sc_list.append(answer_list)
            for j in range(q):
                vote = 0
                for k in range(sc):
                    ans = sc_list[k][j].lower()
                    answer.append(ans)
                    # ans = os.linesep.join([s for s in ans.splitlines() if s]).replace(' ', '')
                    if (
                        "the answer is yes" in ans
                        or "there is a path between node "
                        + str(qt[j][0])
                        + " and node "
                        + str(qt[j][1])
                        in ans
                    ):
                        vote += 1
                if vote * 2 >= sc:
                    res.append(1)
                else:
                    res.append(0)

            for j in range(q):
                vote = 0
                for k in range(sc):
                    ans = sc_list[k][j + q].lower()
                    answer.append(ans)
                    # ans = os.linesep.join([s for s in ans.splitlines() if s]).replace(' ', '')
                    if (
                        "the answer is yes" not in ans
                        and "there is a path between node "
                        + str(qt[q + j][0])
                        + " and node "
                        + str(qt[q + j][1])
                        not in ans
                    ):
                        vote += 1
                if vote * 2 >= sc:
                    res.append(1)
                else:
                    res.append(0)

    res = np.array(res)
    answer = np.array(answer)
    print((res == 1).sum())
    log(Q_list, res, answer, args)


if __name__ == "__main__":
    main()

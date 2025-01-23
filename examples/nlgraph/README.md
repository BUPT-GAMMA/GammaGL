# Can Language Models Solve Graph Problems in Natural Language?

- Paper link: [https://arxiv.org/abs/2305.10037](https://arxiv.org/abs/2305.10037)

- Author's code:[https://github.com/Arthur-Heng/NLGraph]

## Dataset

download the dataset NLGraph from [https://github.com/Arthur-Heng/NLGraph]

## How to run

set your openai key OEPNAI_API_KEY (and openai organization OPENAI_ORGANIZATION optionally):

```bash
$env:OPENAI_API_KEY="your openai key" # for Windows powershell
export OPENAI_API_KEY="your openai key" # for Linux
```

Or use API of Domestic Large Models like tongyiqianwen
set your DASHSCOPE_API_KEY

then run the evaluation code for a specific task:

```python
TL_BACKEND='torch' python evaluation/<task_name>.py --model <name of LM> --mode <difficulty_mode> --prompt <prompting technique> --T <temperature> --token <max number of token> --SC <whether to use self-consistency> --SC_num <sampling number for SC>
```

For instance,

```python
TL_BACKEND='torch' python evaluation/cycle.py --model qwen-max --mode easy --prompt CoT --SC 1 --SC_num 5
```

evaluates qwen-max model on the easy subset of cycle task, using chain-of-thought prompting together with self-consistency.

## Results

| Metric | Connectivity | Cycle | Shortest Path |
| ------ | ------------ | ----- | ------------- |
| Paper  | 94.32        | 80.00 | 63.89         |
| Our    | 95.94        | 50.00 | 55.00         |

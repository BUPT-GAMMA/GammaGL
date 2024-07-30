# Graph Fairness Learning under Distribution Shifts     

- Paper link: [https://arxiv.org/abs/2401.16784](https://arxiv.org/abs/2401.16784)
- Author's code repo: [https://github.com/BUPT-GAMMA/FatraGNN](https://github.com/BUPT-GAMMA/FatraGNN). Note that the original code is implemented with Torch for the paper. 

# Dataset Statics


| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Bail_B0  | 4,686   | 153,942 | 2         |
| Bail_B1  | 2,214   | 49,124  | 2         |
| Bail_B2  | 2,395   | 88,091  | 2         |
| Bail_B3  | 1,536   | 57,838  | 2         |
| Bail_B4  | 1,193   | 30,319  | 2         |
| Credit_C0| 4,184   | 45,718  | 2         |
| Credit_C1| 2,541   | 18,949  | 2         |
| Credit_C2| 3,796   | 28,936  | 2         |
| Credit_C3| 2,068   | 15,314  | 2         |
| Credit_C4| 3,420   | 26,048  | 2         |

Refer to [Credit](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Credit) and [Bail](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Bail).

Results
-------




```bash
TL_BACKEND="torch" python fatragnn_trainer.py --dataset credit --epochs 600 --g_epochs 5 --a_epochs 2 --cla_epochs 12 --dic_epochs 5 --dtb_epochs 5 --c_lr 0.01 --e_lr 0.01
TL_BACKEND="torch" python fatragnn_trainer.py --dataset bail --epochs 400 --g_epochs 5 --a_epochs 4 --cla_epochs 10 --dic_epochs 8 --dtb_epochs 5  --c_lr 0.005 --e_lr 0.005


TL_BACKEND="tensorflow" python fatragnn_trainer.py --dataset credit --epochs 600 --g_epochs 5 --a_epochs 2 --cla_epochs 12 --dic_epochs 5 --dtb_epochs 5 --c_lr 0.01 --e_lr 0.01
TL_BACKEND="tensorflow" python fatragnn_trainer.py --dataset bail --epochs 400 --g_epochs 5 --a_epochs 4 --cla_epochs 10 --dic_epochs 8 --dtb_epochs 5  --c_lr 0.005 --e_lr 0.005
```
ACC:
| Dataset    | Paper       | Our(torch)       | Our(tensorflow) |
| ---------- | ----------- | ---------------- | --------------- |
| Credit_C1  | 77.31±0.10  | 77.08(±0.08)     | 77.06(±0.10)    |
| Credit_C2  | 77.12±0.28  | 77.26(±0.13)     | 77.22(±0.11)    |
| Credit_C3  | 71.81±0.39  | 70.86(±0.15)     | 71.02(±0.12)    |
| Credit_C4  | 72.15±0.42  | 70.91(±0.10)     | 71.08(±0.09)    |
| Bail_B1    | 74.59±0.93  | 72.13(±0.97)     | 72.08(±0.98)    |
| Bail_B2    | 70.46±0.44  | 78.55(±0.94)     | 79.02(±0.31)    |
| Bail_B3    | 71.65±4.65  | 79.77(±0.70)     | 78.96(±0.76)    |
| Bail_B4    | 72.59±3.39  | 80.35(±1.73)     | 79.91(±0.64)    |




equality：
| Dataset    | Paper      | Our(torch)       | Our(tensorflow) |
| ---------- | ---------- | ---------------- | --------------- |
| Credit_C1  | 0.71±0.03  | 0.53(±0.05)      | 0.41(±0.02)     |
| Credit_C2  | 0.95±0.7   | 0.13(±0.10)      | 0.30(±0.39)     |
| Credit_C3  | 0.81±0.56  | 1.81(±1.68)      | 2.51(±1.92)     |
| Credit_C4  | 1.16±0.13  | 0.14(±0.07)      | 0.18(±0.13)     |
| Bail_B1    | 2.38±3.19  | 4.38(±2.87)      | 1.28(±1.04)     |
| Bail_B2    | 0.43±1.14  | 4.48(±2.52)      | 3.51(±1.92)     |
| Bail_B3    | 2.43±4.94  | 2.62(±2.55)      | 2.13(±0.43)     |
| Bail_B4    | 2.45±6.67  | 1.16(±1.40)      | 3.03(±1.22)     |

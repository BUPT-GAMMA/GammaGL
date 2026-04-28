import pandas as pd
import matplotlib.pyplot as plt
import os

# ====================== 只需改这里 ======================
CSV_FILE = "save/citeseer/log.csv"  # 已帮你写对真实路径！
SAVE_DIR = "images"                 # 自动保存到 images
# ========================================================

# 创建图片文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取 CSV
df = pd.read_csv(CSV_FILE, sep=",", skipinitialspace=True)

# 清理列名（去空格）
df.columns = [col.strip() for col in df.columns]

# 按模型画曲线
models = df["Model"].unique()
plt.figure(figsize=(12, 7))

for model in models:
    sub = df[df["Model"] == model]
    acc = sub["Acc"].values
    plt.plot(acc, marker='o', linewidth=2, label=model)

# 画图样式
plt.xlabel("Experiment Order", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(f"Accuracy Curve by Model", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存
save_path = os.path.join(SAVE_DIR, "citeseer_acc.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ 图片已保存到：{save_path}")
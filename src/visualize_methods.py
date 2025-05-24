
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'  # 柔らかい丸ゴシックえ

def visualize(csv_pth):
    
    df = pd.read_csv(csv_pth, index_col=0)
    print(df)
    
    
    plt.rcParams['figure.subplot.left'] = 0.3
    svm = sns.heatmap(df, annot=True, cmap='coolwarm')
    
    figure = svm.get_figure()    
    timestr = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    figure.savefig(os.path.join(Path(__file__).parents[1], f'output/imgs/svm_conf_{timestr}.png'), dpi=400)



def plot_political_bubble(df, save_path="output/political_bubble_plot.png"):
    """
    Presenterごとの右派−左派のバブルプロットを描画。
    """
    # X: スコア差分、Y: サンプル数、サイズ: 極性強度
    df["ScoreDiff"] = df["Avg_Right_Score"] - df["Avg_Left_Score"]
    df["PolarityStrength"] = df[["Avg_Right_Score", "Avg_Left_Score"]].max(axis=1) * 100  # 可視性のため倍率調整

    plt.figure(figsize=(10, 6))
    df["PolarityStrength"] = df[["Avg_Right_Score", "Avg_Left_Score"]].max(axis=1)
    df["BubbleSize"] = np.sqrt(df["PolarityStrength"]) * 1000  # スケーリング＋視認性確保

    scatter = plt.scatter(
        df["ScoreDiff"],
        df["SampleCount"],
        s=df["BubbleSize"],
        alpha=0.6,
        c=df["ScoreDiff"],
        cmap="bwr",
        edgecolors="k"
    )
    

    # ラベル表示
    for i, row in df.iterrows():
        label = f"{row['Presenter']} ({row['Role']})"
        plt.text(row["ScoreDiff"], row["SampleCount"], label, fontsize=8, ha='center', va='bottom')

    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Right-wing ←→ Left-wing (Score Difference)")
    plt.ylabel("Sample Count")
    plt.title("Presenter Political Leaning Bubble Plot")
    plt.grid(True)
    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path

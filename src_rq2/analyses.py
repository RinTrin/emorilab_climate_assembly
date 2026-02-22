# analyses.py
import os
import re
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import pearsonr

import torch

# 安定化（CPU暴走防止 & 並列トークナイズ抑制）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.set_num_threads(2)
except Exception:
    pass

def _use_japanese_font():
    # macOSに入っている日本語フォントの候補
    candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            name = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # フォールバック（Noto/IPAが入っていれば自動検出）
    for f in fm.findSystemFonts():
        if any(s in f for s in ["NotoSansCJK", "Noto Sans CJK", "IPAexGothic"]):
            name = fm.FontProperties(fname=f).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return

_use_japanese_font()


# --- YAMLキーと照合させるための最小正規化 ---
def _canon_source_key(x: str) -> str:
    """
    例:
      'lecture4B_6_1_merged_segmented.pkl' -> 'lecture4b_6_1'
      'Lecture4B_7.txt'                   -> 'lecture4b_7'
    処理:
      - ベース名のみ
      - 多重拡張子/既知サフィックス（_segmented/_merged/_punc_added/_youtube_txt/_txt）を剥がす
      - 'lecture'で始まる英数とアンダースコアのみを抽出
      - 小文字化
    """
    b = os.path.basename(str(x))
    stem = b
    # 多重拡張子を最大3回まで剥がす
    for _ in range(3):
        s, ext = os.path.splitext(stem)
        if not ext:
            break
        stem = s
    for suf in ("_segmented", "_merged", "_punc_added", "_youtube_txt", "_txt"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    m = re.search(r"(?i)(lecture[a-z0-9_]+)", stem)  # ignore case
    key = m.group(1) if m else stem
    return key.lower()




def actor_analysis(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname):
    # Summarize by source and role
    summarize_top1_by_source_pth, summarize_top1_by_role_pth = summarize(analyzed_csv_pth, presenter_role_dict)

    # Visualize day by day
    day_by_day_output_path = f"/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/imgs/plot_presenter_data_day_by_day_{actionplan_excel_sheetname}.png"
    plot_presenter_data_day_by_day(summarize_top1_by_source_pth, output_path=day_by_day_output_path)
    boxplot_output_path = f"/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/imgs/role_boxplot_{actionplan_excel_sheetname}.png"
    plot_role_boxplot(summarize_top1_by_source_pth, boxplot_output_path)


def summarize(csv_path, presenter_role_dict):
    df = pd.read_csv(csv_path)
    return summarize_top1_by_source(df, presenter_role_dict,
        save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files"
    )[1], summarize_top1_by_role(
        summarize_top1_by_source(df, presenter_role_dict,
            save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files"
        )[0],
        save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files"
    )[1]


def summarize_top1_by_source(df, presenter_role_dict, save_dir="."):
    """
    表1：Top1_SourceFile別の出現割合を計算し、PresenterとRoleを付加してCSV出力。
    """
    # 正規化キーで集計（YAMLと必ず一致させる）
    presenter_role_dict_lc = {str(k).lower(): v for k, v in presenter_role_dict.items()}
    df = df.copy()
    df["SourceKey"] = df["Top1_SourceFile"].apply(_canon_source_key)

    source_counts = df["SourceKey"].value_counts().reset_index()
    source_counts.columns = ["SourceFile", "Count"]

    total = source_counts["Count"].sum()
    source_counts["Percentage"] = source_counts["Count"] / total * 100

    source_counts["Presenter"] = source_counts["SourceFile"].map(
        lambda k: presenter_role_dict_lc.get(str(k).lower(), {}).get("Presenter", "Unknown")
    )
    source_counts["Role"] = source_counts["SourceFile"].map(
        lambda k: presenter_role_dict_lc.get(str(k).lower(), {}).get("Role", "Unknown")
    )

    summary_df = source_counts[["SourceFile", "Presenter", "Role", "Percentage"]]

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "top1_sourcefile_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return summary_df, csv_path


def summarize_top1_by_role(summary_df, save_dir="."):
    """
    表2：RoleごとのTop1出現割合と、対応するPresenter一覧を含む表を生成してCSV保存。
    """
    # Percentage集計
    role_summary = summary_df.groupby("Role")["Percentage"].sum().reset_index()
    role_summary.columns = ["Role", "TotalPercentage"]

    # PresenterListを作成
    role_to_presenters = summary_df.groupby("Role")["Presenter"].unique().reset_index()
    role_to_presenters.columns = ["Role", "PresenterList"]

    # PresenterListを文字列に変換（見やすくする）
    role_to_presenters["PresenterList"] = role_to_presenters["PresenterList"].apply(
        lambda x: ", ".join(sorted(set(x)))
    )

    # 結合
    merged = pd.merge(role_summary, role_to_presenters, on="Role")

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "top1_role_summary.csv")
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return merged, csv_path

def presentation_length_analysis(
    csv_path,
    presenter_role_dict,
    save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results",
    role_colors=None,
    actionplan_excel_sheetname=None
):
    """
    固定プレゼン時間(分) × Top1参照シェア(%) を可視化。
    - Roleで色分け、ラベルは alpha=0.8 / rotation=30°
    - フォントはあなたの `_use_japanese_font()` に完全委譲（本関数では一切上書きしない）
    依存: _canon_source_key()。実行前に _use_japanese_font() が呼ばれていること。
    """
    _use_japanese_font()

    # データ
    df = pd.read_csv(csv_path)
    if "Top1_SourceFile" not in df.columns:
        raise ValueError("CSVに 'Top1_SourceFile' がありません。")

    keys = df["Top1_SourceFile"].apply(_canon_source_key)
    vc = keys.value_counts()
    pct = keys.value_counts(normalize=True) * 100
    counts = pd.DataFrame({"SourceFile": vc.index, "Count": vc.values, "Percentage": pct.values})

    # YAMLマッピング
    m = {str(k).lower(): v for k, v in presenter_role_dict.items()}
    def get(k, field, default=np.nan):
        return m.get(str(k).lower(), {}).get(field, default)

    counts["Presenter"] = counts["SourceFile"].map(lambda k: get(k, "Presenter", "Unknown"))
    counts["Role"] = counts["SourceFile"].map(lambda k: get(k, "Role", "Unknown"))
    counts["PresentationLengthSecond"] = counts["SourceFile"].map(lambda k: get(k, "PresentationLengthSecond"))
    counts = counts.dropna(subset=["PresentationLengthSecond"]).copy()
    counts["PresentationLengthMinute"] = counts["PresentationLengthSecond"] / 60

    # 相関
    if len(counts) >= 2:
        r, p = pearsonr(counts["PresentationLengthSecond"].astype(float), counts["Percentage"].astype(float))
        r2 = r**2
    else:
        r2, p = np.nan, np.nan

    # 色
    role_colors = role_colors or {
        "academic": "#1f77b4",
        "citizen":  "#2ca02c",
        "private":  "#ff7f0e",
        "public":   "#d62728",
        "Unknown":  "gray",
    }

    # 可視化（rcParamsのフォント設定に完全依存）
    out_dir = os.path.join(save_dir, "presentation_length_analysis")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for role, grp in counts.groupby("Role"):
        xs = grp["PresentationLengthMinute"].to_numpy()
        ys = grp["Percentage"].to_numpy()
        ax.scatter(xs, ys, s=100, alpha=0.8, color=role_colors.get(role, "gray"),
                   label=role, edgecolors="white", linewidths=0.8)
        for _, row in grp.iterrows():
            ax.text(row["PresentationLengthMinute"],
                    row["Percentage"] + 0.15,
                    str(row.get("Presenter", "")),
                    color="black", ha="center", fontsize=8, rotation=30)

    if len(counts) >= 2:
        x = counts["PresentationLengthMinute"].to_numpy()
        y = counts["Percentage"].to_numpy()
        k, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, k * xline + b, linewidth=2)

    ax.set_xlabel("プレゼン時間（分）")
    ax.set_ylabel("アクションプランへの参照率（%）")
    ax.set_title("プレゼン時間とアクションプランへの参照率の関係（役割別）")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Role", loc="best", frameon=True)

    # 注記
    txt = f"R² = {r2:.2f}\nP = {p:.3f}" if np.isfinite(r2) else "R² = N/A\nP = N/A"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"reference_share_vs_presentation_length_colored_{actionplan_excel_sheetname}.png")
    csv_out = os.path.join(out_dir, f"reference_share_vs_presentation_length_{actionplan_excel_sheetname}.csv")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    counts.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"[✔] グラフ保存: {plot_path}")
    print(f"[✔] 集計CSV保存: {csv_out}")
    if np.isfinite(r2):
        print(f"[✔] R²={r2:.3f}, p={p:.3f}")


def plot_presenter_data_day_by_day(
    csv_path, 
    output_path=None
):
    df = pd.read_csv(csv_path)

    def extract_day(source):
        m = re.search(r"lecture(\d+)", str(source))
        return int(m.group(1)) if m else None

    df["Day"] = df["SourceFile"].apply(extract_day)

    role_colors = {
        "academic": "#1f77b4",
        "citizen": "#2ca02c",
        "private": "#ff7f0e",
        "public": "#d62728",
    }

    day_labels = sorted([d for d in df["Day"].unique() if d is not None])
    day_mapping = {day: i for i, day in enumerate(day_labels, start=1)}
    df["DayIndex"] = df["Day"].map(day_mapping)

    plt.figure(figsize=(10, 6))

    for role, group in df.groupby("Role"):
        plt.scatter(
            group["DayIndex"],
            group["Percentage"],
            color=role_colors.get(role, "gray"),
            s=100,
            label=role,
            alpha=0.8,
        )
        for _, row in group.iterrows():
            if pd.isna(row["DayIndex"]):
                continue
            plt.text(
                row["DayIndex"] + 0.1,
                row["Percentage"],
                row.get("Presenter", ""),
                color="black",
                ha="left",
                fontsize=8,
                rotation=0,
            )

    plt.xticks(ticks=list(day_mapping.values()), labels=[f"{i}日目" for i in day_labels])
    plt.ylabel("アクションプランへの参照率（%）")
    plt.xlabel("情報提供日")
    plt.title("情報提供日とアクションプランへの参照率の関係（役割別）")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Role", loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_role_boxplot(
    csv_path: str,
    output_path: str = ""
):
    # データ読み込み
    df = pd.read_csv(csv_path)

    # スタイルと色設定
    sns.set(style="whitegrid")
    bright_blue = "#4A90E2"

    # 描画
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="Percentage",
        y="Role",
        data=df,
        color=bright_blue,
        orient="h",
        linewidth=1.5,
        fliersize=3,
    )

    # 日本語フォント（環境依存。存在しない場合はスキップ）
    try:
        import matplotlib.font_manager as fm

        font_path = "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.title("Role別 割合の箱ひげ図", fontsize=14, fontproperties=font_prop)
        else:
            plt.title("Role別 割合の箱ひげ図", fontsize=14)
    except Exception:
        plt.title("Role別 割合の箱ひげ図", fontsize=14)

    plt.xlabel("Percentage (%)", fontsize=12)
    plt.ylabel("Role", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
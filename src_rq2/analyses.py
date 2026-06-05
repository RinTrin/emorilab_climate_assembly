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
from pathlib import Path

import torch
from datetime import datetime

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
    YAMLキーと matched_input_pkl を照合するための正規化。
    """
    b = os.path.basename(str(x))
    stem = b.lower()

    # 多重拡張子を最大3回まで剥がす
    for _ in range(3):
        s, ext = os.path.splitext(stem)
        if not ext:
            break
        stem = s

    suffixes = [
        "_youtube_txt_punc_added_syusei",
        "_youtube_txt_punc_added",
        "_youtube_txt_syusei",
        "_punc_added_syusei",
        "_punc_added",
        "_youtube_txt",
        "_segmented",
        "_merged",
        "_syusei",
        "_txt",
    ]

    # 複数 suffix が連結していても剥がしきる
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
                break

    m = re.search(r"(lecture[a-z0-9_]+)", stem)
    key = m.group(1) if m else stem

    return key.lower()




def actor_analysis(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname, city_name):
    # Summarize by source and role
    summarize_top1_by_source_pth, summarize_top1_by_role_pth = summarize(analyzed_csv_pth, presenter_role_dict, city_name)

    # Visualize day by day
    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    day_by_day_output_path = f"/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/imgs/plot_presenter_data_day_by_day_{actionplan_excel_sheetname}_{city_name}_{timestr}.png"
    plot_presenter_data_day_by_day(summarize_top1_by_source_pth, output_path=day_by_day_output_path)
    boxplot_output_path = f"/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/imgs/role_boxplot_{actionplan_excel_sheetname}_{city_name}_{timestr}.png"
    plot_role_boxplot(summarize_top1_by_source_pth, boxplot_output_path)


def summarize(csv_path, presenter_role_dict, city_name):
    df = pd.read_csv(csv_path)
    return summarize_top1_by_source(df, presenter_role_dict,
        save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files",city_name=city_name
    )[1], summarize_top1_by_role(
        summarize_top1_by_source(df, presenter_role_dict,
            save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files",city_name=city_name
        )[0],
        save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files"
    )[1]


def summarize_top1_by_source(df, presenter_role_dict, save_dir=".", city_name=None):
    """
    matched_input_pkl別の出現割合を計算し、PresenterとRoleを付加してCSV出力。
    参照されていない presenter も Count=0, Percentage=0 として含める。
    """

    df = df.copy()
    df = df[df["similar_check"] == True]
    df = df[df["city_name"] == city_name]
    df["SourceKey"] = df["matched_input_pkl"].apply(_canon_source_key)

    # 1. 参照されたSourceKeyを集計
    source_counts = df["SourceKey"].value_counts().reset_index()
    source_counts.columns = ["SourceKey", "Count"]
    total = source_counts["Count"].sum()

    # 2. YAML順を保持
    yaml_order_map = {
        _canon_source_key(source_key): i
        for i, source_key in enumerate(presenter_role_dict.keys())
    }

    # 3. YAML側から全 presenter を作る
    presenter_rows = []
    for source_key, info in presenter_role_dict.items():
        canon_key = _canon_source_key(source_key)
        presenter_rows.append({
            "SourceKey": canon_key,
            "Presenter": info.get("Presenter", "Unknown"),
            "Role": info.get("Role", "Unknown"),
            "YamlOrder": yaml_order_map.get(canon_key, 10**9),
        })

    all_presenters = pd.DataFrame(presenter_rows).drop_duplicates(subset=["SourceKey"])

    # 4. left join
    summary_df = all_presenters.merge(
        source_counts,
        on="SourceKey",
        how="left"
    )

    summary_df["Count"] = summary_df["Count"].fillna(0).astype(int)
    summary_df["Percentage"] = summary_df["Count"] / total * 100 if total > 0 else 0.0

    # 5. 日付順 + YAML順
    summary_df["Day"] = summary_df["SourceKey"].str.extract(r"lecture(\d+)").astype(float)
    summary_df = summary_df.sort_values(["Day", "YamlOrder"]).drop(columns=["Day"])

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "matched_input_pkl_summary.csv")
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
    actionplan_excel_sheetname=None,
    city_name=None
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
    df = df.copy()
    df = df[df["similar_check"] == True]   # 追加
    df = df[df["city_name"] == city_name]
    if "matched_input_pkl" not in df.columns:
        raise ValueError("CSVに 'matched_input_pkl' がありません。")

    keys = df["matched_input_pkl"].apply(_canon_source_key)
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

    df["Day"] = df["SourceKey"].apply(extract_day)

    role_colors = {
        "academic": "#1f77b4",
        "citizen": "#2ca02c",
        "private": "#ff7f0e",
        "public": "#d62728",
        "Unknown": "gray",
    }

    day_labels = sorted([d for d in df["Day"].unique() if pd.notna(d)])
    day_mapping = {day: i for i, day in enumerate(day_labels, start=1)}
    df["DayIndex"] = df["Day"].map(day_mapping)

    # YAML順で並べる
    if "YamlOrder" not in df.columns:
        raise ValueError("YamlOrder列がありません。summarize_top1_by_source() 側で付与してください。")

    df = df.sort_values(["DayIndex", "YamlOrder"]).copy()

    # 同じ日内で左から右へ番号を振る
    df["OrderInDay"] = df.groupby("DayIndex").cumcount()
    df["NInDay"] = df.groupby("DayIndex")["SourceKey"].transform("count")

    # ずらし幅は狭める
    # 総幅が広がりすぎないように、1日あたりの最大総幅を0.24に抑える
    max_total_span = 0.24

    def calc_step(n):
        if n <= 1:
            return 0.0
        return min(0.04, max_total_span / (n - 1))

    df["JitterStep"] = df["NInDay"].apply(calc_step)

    df["PlotX"] = df["DayIndex"] + (
        df["OrderInDay"] - (df["NInDay"] - 1) / 2
    ) * df["JitterStep"]

    plt.figure(figsize=(12, 7))

    for role, group in df.groupby("Role"):
        plt.scatter(
            group["PlotX"],
            group["Percentage"],
            color=role_colors.get(role, "gray"),
            s=100,
            label=role,
            alpha=0.8,
        )

        for _, row in group.iterrows():
            if pd.isna(row["PlotX"]):
                continue

            label_y = row["Percentage"] + 0.25

            plt.text(
                row["PlotX"],
                label_y,
                row.get("Presenter", ""),
                color="black",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=30,
            )

    plt.xticks(
        ticks=list(day_mapping.values()),
        labels=[f"{i}日目" for i in day_labels]
    )

    ymax = df["Percentage"].max()
    plt.ylim(bottom=-1, top=max(5, ymax * 1.15))

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


def create_object_role_percentage_table(csv_path: str | Path, output_path: str | Path = None) -> pd.DataFrame:
    """
    CSVから、アクションの客体（object）と講義者属性（matched_role）の
    対応表を作成する。

    縦軸:
        行政、市民、事業者

    横軸:
        private, citizen, public, academic

    割合の分母:
        matched_roleごとのアクション総数

    複数主体の扱い:
        objectが「市民・行政」のように複数主体を含む場合、
        該当するすべての主体に重複して計上する。

    Parameters
    ----------
    csv_path : str | pathlib.Path
        object列とmatched_role列を含むCSVファイルのパス。

    Returns
    -------
    pandas.DataFrame
        各セルを「xx.x%」形式で表示した対応表。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_columns = {"object", "matched_role"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSVに必要な列がありません: {sorted(missing_columns)}"
        )

    object_order = ["行政", "市民", "事業者"]
    role_order = ["private", "citizen", "public", "academic"]

    # matched_roleの表記揺れを抑える
    df["matched_role"] = (
        df["matched_role"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # object列を分割する。
    # 「・」以外の区切り文字が混ざった場合にもある程度対応する。
    def split_objects(value: object) -> list[str]:
        if pd.isna(value):
            return []

        objects = re.split(r"[・、,／/|]+", str(value))

        return [
            item.strip()
            for item in objects
            if item.strip() in object_order
        ]

    df["object_list"] = df["object"].apply(split_objects)

    # 複数主体を別々の行として展開する。
    # 例: 「事業者・行政」→「事業者」と「行政」の2行
    exploded_df = df.explode("object_list")

    count_table = pd.crosstab(
        index=exploded_df["object_list"],
        columns=exploded_df["matched_role"],
    )

    # 行列の順序を固定する。
    # CSV内に該当データがない属性も0として表示する。
    count_table = count_table.reindex(
        index=object_order,
        columns=role_order,
        fill_value=0,
    )

    # 分母は、各matched_roleに対応する元のアクション数。
    # 重複展開前の行数を使う。
    denominator = (
        df["matched_role"]
        .value_counts()
        .reindex(role_order, fill_value=0)
    )

    percentage_table = count_table.div(
        denominator.replace(0, pd.NA),
        axis="columns",
    )

    formatted_table = percentage_table.map(
        lambda value: "-" if pd.isna(value) else f"{value:.1%}"
    )

    formatted_table.index.name = "object"
    formatted_table.columns.name = "matched_role"
    
    if output_path:
        formatted_table.to_csv(os.path.join(output_path, "object_percentage.csv"), encoding="utf-8-sig")

    return formatted_table


def create_city_object_share_table(
    csv_path: str | Path,
    output_path: str | Path = None
) -> pd.DataFrame:
    """
    市ごとに、object列に記載された主体の構成比を集計する。

    複数主体が記載されている場合は、1行分の重みを均等配分する。
    例:
        「行政」           -> 行政: 1.0
        「行政・市民」     -> 行政: 0.5, 市民: 0.5
        「行政・市民・事業者」 -> 各主体: 1/3

    各市の横方向の合計は100%になる。

    Parameters
    ----------
    csv_path : str | pathlib.Path
        city_name列とobject列を含むCSVファイルのパス。

    Returns
    -------
    pandas.DataFrame
        市ごとの主体構成比を「xx.x%」形式で表示した表。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_columns = {"city_name", "object"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSVに必要な列がありません: {sorted(missing_columns)}"
        )

    object_order = ["行政", "市民", "事業者"]

    def split_objects(value: object) -> list[str]:
        if pd.isna(value):
            return []

        items = re.split(r"[・、,／/|]+", str(value))

        # 同じ主体が重複記載されていても1回だけ数える
        return list(
            dict.fromkeys(
                item.strip()
                for item in items
                if item.strip() in object_order
            )
        )

    df["object_list"] = df["object"].apply(split_objects)

    # objectが空欄または想定外の表記になっている行を検出
    invalid_rows = df[df["object_list"].map(len) == 0]

    if not invalid_rows.empty:
        raise ValueError(
            "object列から主体を判定できない行があります。"
            f" 該当行数: {len(invalid_rows)}"
        )

    # 1行の合計が必ず1になるように均等配分
    for target_object in object_order:
        df[target_object] = df["object_list"].apply(
            lambda objects: (
                1 / len(objects)
                if target_object in objects
                else 0
            )
        )

    share_table = (
        df.groupby("city_name", sort=False)[object_order]
        .mean()
    )

    formatted_table = share_table.map(
        lambda value: f"{value:.1%}"
    )

    formatted_table.index.name = "city_name"
    
    formatted_table.to_csv(
        os.path.join(output_path, "city_object_percentage_table.csv"),
        encoding="utf-8-sig",
    )

    return formatted_table
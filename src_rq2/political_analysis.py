import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from calc_similarity_bert import SentenceBertJapanese
import umap
import torch.nn.functional as F
from bertopic import BERTopic
import seaborn as sns

import torch
from utils import pickle_load, input_organize, _to_np32, _finite, generate_reference_data_climate_cached

def political_analysis(analyzed_csv_pth, presenter_role_dict, input_materials_pkl_pth_list, actionplan_excel_sheetname):
    """
    類似度解析結果CSVを受け取り、政治的傾向の推定と可視化を実行
    - 横軸: ScoreRel（平均との差＝相対位置のみ）
    - 縦軸: RefCount（Top1として参照された回数）
    """
    political_dir = os.path.join(
        "/Users/rintrin/codes/emorilab_climate_assembly/analysis_results", "political_analysis"
    )
    os.makedirs(political_dir, exist_ok=True)

    # --- Top1参照回数（縦軸） ---
    df_csv = pd.read_csv(analyzed_csv_pth)
    refcount = (
        df_csv["matched_input_pkl"]
        .value_counts()
        .rename_axis("Expert")
        .reset_index(name="RefCount")
    )

    # --- 参照された文だけをベクトル化（Top1） ---
    opinion_vector_for_each_expert = get_opinion_vector(analyzed_csv_pth, presenter_role_dict)

    # ベクトルの平均化
    average_vector_for_each_expert = get_average_opinion_vector(opinion_vector_for_each_expert)

    # --- 参照軸データ ---
    reference_data_from_article = {
        "right_texts": ["猛暑になるたびに「地球温暖化のせいだ」とよく報道される。しかし、本当だろうか？例えば、2018年の夏は猛暑だった。そして、政府資料を見ると「東日本の7月の平均気温が平年より2.8℃高くなり、これは1946年の統計開始以降で第1位の高温であった」、「熊谷で最高気温が国内の統計開始以来最高となる41.1℃になった」とし、その原因には「地球温暖化の影響があった」としている（政府資料「はじめに」）。だが、全国津々浦々の気象観測施設に足を運び、測定に悪影響を与える周辺環境を点検、改善策を提案していることから「気象観測の水戸黄門」と呼ばれる東北大学の近藤純正名誉教授による推計では、日本の平均気温の上昇速度は100年あたり0.89℃程度、過去30年でわずか0.3℃だった。つまり、「熊谷で41.1℃になった」が、これへの地球温暖化の寄与は、もし過去30年間に地球温暖化がなければ40.8℃であったということだ。地球温暖化は、ごくわずかに温度を上げているに過ぎない。平均気温についても同じようなことがいえる。政府発表で東日本の2018年7月の平均気温が平年より2.8℃高かったとしているが、これも、もし過去30年間に地球温暖化がなければ2.5℃高かったということだ。猛暑であることに変わりはない。猛暑をもたらした最大の要因は自然変動だ。自然変動とは、具体的にいうと、太平洋高気圧の張り出しといった気圧配置の変化、ジェット気流の蛇行、梅雨前線の活動の強弱やタイミングなどである。つまりは、天気図で日々、われわれが目にする気象の変化のことである。このような自然変動で起きる気温変化とは、地球規模で見てどのようなものなのか図表1を見るとよくわかる。この図表では、2023年の気温が直近の10年間の平均と比較されている。これを見ると、日本では、2023年は暑かった（緑色）ことがわかる。だがその一方で、アフリカ、インド、中国などは寒かった（青色）こともわかる。"],
        "left_texts": ["このままいけば地球環境は取り返しのつかない状態になり、社会の分断はますます進んでいくでしょう。『人新世の「資本論」』ではそうした状態を「気候ファシズム」と呼んでいます。そうならないようにするためにも、2020年代を時代のひとつの分岐点と考えるべきです。今、多くの科学者たちは「一般的に最悪の気候変動の帰結を防ぐには、2100年までの気温上昇を産業革命以前に比べて、1.5度以内に抑えないといけない」と言っています。そのためには、2050年を見据えた「脱炭素化」の目標に向けて、まずは2030年までに二酸化炭素（CO2）排出量をほぼ半減にする必要があります。2020年はコロナの影響でCO2排出量が前年比5.8％減少したものの、すでにCO2の排出量は再びコロナ禍以前よりも増え始めています。このままでは、2030年には科学者が警告する1.5度を超える可能性があります。「1.5度という基準を2030年で超え、気候ファシズムに進んでいくのか」、あるいは「2100年までの気温上昇を１．５度に抑えるため、世界で一致団結する方向に舵を切るのか」、その分岐点となるのが2020年代だと言えます。菅政権も2050年に向けて「脱炭素社会化の実現」を掲げ、大企業も動き始めています。ただ、日本の施策で気になっていることは、菅政権が発表した2050年の目標が「産業政策」としての脱炭素化になっていることです。つまり、再生可能エネルギーや電気自動車に積極的に投資することによって、その部門で世界をけん引するような技術革新を起こす「緑の経済成長」として、この問題が捉えられている点です。「緑の経済成長」は根本的な問題解決になるのでしょうか。私は解決にならないと考えています。もし、電気自動車で問題が解決するのであれば「自動車を作ればいい」と考えるでしょう。しかしそのために南米やアフリカからEV電池用のリチウムやコバルトを獲得してしまうため、結果的に自然環境を破壊しています。先進国だけが電気自動車に乗って、今までと変わらない快適な生活を続けるでしょう。こうして、これまでもさんざん語られてきた「南北問題」──「グローバルサウス」と「グローバルノース」の格差が広がり、グローバルサウスに負担が押し付けられてしまいます。基本的には、20世紀に繰り広げられてきた「植民地主義」や「帝国主義」がそのまま繰り返されることになります。　「SDGsで格差をなくそう」という理念を掲げながら、今までと同じような経済成長を求めるシステムでは、別の形での格差や不公正、不平等などがくり返し再生産されてしまいます。これでは、根本的な問題解決にはなりません。気候変動問題やパンデミック問題を契機に、今までの私たちの生活の豊かさの背後にある抑圧や暴力性を見直し、是正していかなければなりません。しかし、それらを技術や経済成長のネタにしてしまうことで根本的な問題から目をそらしています。つまり、無限の経済成長を求める資本主義が原因だということを認識しないといけません。資本主義から決別し、まったく別の社会を構築できるかが、今こそ問われていることなのです。"]
    }
    reference_data_from_gpt = generate_reference_data_climate_cached()

    for mode, reference_data in [["article", reference_data_from_article], ["gpt", reference_data_from_gpt]]:
        axis_vector = create_reference_axis(reference_data=reference_data)
        df_scores = project_vectors_to_axis(average_vector_for_each_expert, axis_vector)

        # 0に意味はないので相対化（平均との差）
        df_scores["ScoreRel"] = df_scores["Score"] - df_scores["Score"].mean()

        # RefCount を付与
        df_plot = df_scores.merge(refcount, on="Expert", how="left")
        df_plot["RefCount"] = df_plot["RefCount"].fillna(0).astype(int)

        # scatter（横=相対スコア, 縦=参照回数）
        plot_political_scatter(
            df_plot,
            presenter_role_dict,
            save_dir=political_dir,
            file_name=f"{mode}_{actionplan_excel_sheetname}"
        )

        # CSV保存
        df_plot.to_csv(os.path.join(political_dir, f"df_scores_{mode}_{actionplan_excel_sheetname}.csv"), index=False)


def get_opinion_vector(analyzed_csv_pth: str, presenter_role_dict: dict) -> dict:
    """
    analyzed_csv の Top1 の参照文だけをSBERTでembeddingし、
    Source から `_youtube_txt_segmented.pkl` を除去して得た yaml_key 単位でまとめて返す。
    戻り値: {yaml_key: (N_i, dim) np.ndarray}
    """
    df = pd.read_csv(analyzed_csv_pth)

    SRC_COL = "Top1_SourceFile"
    SENT_COL = "Top1_Text"

    if SRC_COL not in df.columns or SENT_COL not in df.columns:
        raise ValueError(f"CSV列が想定と違います: need {SRC_COL}, {SENT_COL} / columns={list(df.columns)}")

    # 欠損除外
    df = df[[SRC_COL, SENT_COL]].dropna()
    df[SENT_COL] = df[SENT_COL].astype(str).str.strip()
    df = df[df[SENT_COL] != ""]

    model = SentenceBertJapanese("cl-tohoku/bert-base-japanese-v3")
    buckets = {}  # {yaml_key: [sent, ...]}

    for src, sent in zip(df[SRC_COL].tolist(), df[SENT_COL].tolist()):
        key = str(src).replace("_youtube_txt_segmented.pkl", "")
        # 念のため：もしそのままYAMLに無ければ key のまま進める（unknownになるだけ）
        if key not in presenter_role_dict:
            pass
        buckets.setdefault(key, []).append(sent)

    results = {}
    with torch.no_grad():
        for key, sents in tqdm(buckets.items(), desc="Encoding Top1 referenced sentences"):
            emb = model.encode(sents, batch_size=64)
            emb = F.normalize(torch.as_tensor(emb), p=2, dim=1).cpu().numpy().astype(np.float32)
            results[key] = emb

    return results

def get_average_opinion_vector(opinion_vector_for_each_expert):
    averaged = {}
    for src, vecs in opinion_vector_for_each_expert.items():
        # vecs が list[ndarray] の場合に備えて縦結合
        if isinstance(vecs, list):
            vecs = np.vstack(vecs)
        # torch.Tensor が来た場合も一応ケア
        if torch.is_tensor(vecs):
            vecs = vecs.detach().cpu().numpy()

        # 1文しか無いときの (d,) 対策
        if vecs.ndim == 1:
            avg_vec = vecs.astype(np.float32)
        else:
            avg_vec = vecs.mean(axis=0).astype(np.float32)

        averaged[src] = avg_vec
    return averaged


def create_reference_axis(reference_data):
    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    model = SentenceBertJapanese(MODEL_NAME)

    pro_emb = model.encode(reference_data["right_texts"], batch_size=64)
    con_emb = model.encode(reference_data["left_texts"], batch_size=64)

    # NumPyに統一
    pro_emb = pro_emb.detach().cpu().numpy() if hasattr(pro_emb, "detach") else np.asarray(pro_emb)
    con_emb = con_emb.detach().cpu().numpy() if hasattr(con_emb, "detach") else np.asarray(con_emb)

    # 入力の健全性（空チェック）
    if pro_emb.size == 0 or con_emb.size == 0:
        raise ValueError("create_reference_axis: right_texts/left_texts が空です。参照文を見直してください。")

    pro_vec = pro_emb.mean(axis=0).astype(np.float32)
    con_vec = con_emb.mean(axis=0).astype(np.float32)

    axis = pro_vec - con_vec
    norm = np.linalg.norm(axis)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("create_reference_axis: 基準軸のノルムが0/非有限です。参照文の差が弱すぎます。")
    return axis / norm

def compute_reference_vectors(reference_data, model_name="cl-tohoku/bert-base-japanese-v3"):
    """
    reference_data: {"right_texts": [...], "left_texts": [...]}
    戻り値: {"REF_PRO": vec, "REF_CON": vec} （np.float32, 1D, L2正規化後の平均）
    """
    model = SentenceBertJapanese(model_name)

    pro_emb = model.encode(reference_data["right_texts"], batch_size=64)
    con_emb = model.encode(reference_data["left_texts"], batch_size=64)

    # 行方向L2正規化（SentenceBERTの後処理と同様）
    pro_emb = F.normalize(torch.tensor(pro_emb), p=2, dim=1).cpu().numpy()
    con_emb = F.normalize(torch.tensor(con_emb), p=2, dim=1).cpu().numpy()

    pro_vec = pro_emb.mean(axis=0).astype(np.float32)
    con_vec = con_emb.mean(axis=0).astype(np.float32)

    return {"REF_RIGHT": pro_vec, "REF_LEFT": con_vec}

def _to_np_float32(x):
    import torch
    if isinstance(x, list):
        x = np.array(x)
    if 'torch' in str(type(x)):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(np.float32)

def _is_finite_array(arr):
    return np.isfinite(arr).all()

def project_vectors_to_axis(average_vector_for_each_expert, axis_vector):
    """
    各専門家の平均ベクトルを基準軸に射影してスコア化（cosine射影）
    - axis / v の両方をL2正規化
    - 型/次元/NaN/ゼロを安全にケア
    """
    results = []

    # --- 軸の正規化 ---
    axis = _to_np_float32(axis_vector).reshape(-1)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0 or not np.isfinite(axis_norm):
        raise ValueError("axis_vector がゼロまたは非有限です。基準軸の作成を見直してください。")
    axis = axis / axis_norm

    for name, vec in average_vector_for_each_expert.items():
        v = _to_np_float32(vec).reshape(-1)

        # 次元チェック
        if v.shape[0] != axis.shape[0]:
            print(f"⚠️ 次元不一致のためスキップ: {name} (vec_dim={v.shape[0]}, axis_dim={axis.shape[0]})")
            continue

        if not (_is_finite_array(v) and _is_finite_array(axis)):
            print(f"⚠️ NaN/Inf 検出のためスキップ: {name}")
            continue

        # --- ★追加：v も正規化（論文準拠） ---
        v_norm = np.linalg.norm(v)
        if v_norm == 0 or not np.isfinite(v_norm):
            print(f"⚠️ ゼロ/非有限ベクトルのためスキップ: {name}")
            continue
        v = v / v_norm

        # cosine 射影
        score = float(np.dot(v, axis))
        results.append({"Expert": name, "Score": score})

    if len(results) == 0:
        raise ValueError("射影可能なデータがありません（全件スキップされました）。")

    df = pd.DataFrame(results).sort_values("Score", ascending=False)
    return df

def plot_political_scores_table(df_scores, presenter_role_dict, save_dir, file_name=None):
    """
    射影スコア（右派−左派軸に沿った値）をPresenter・Role付きで一覧化して画像出力
    """
    os.makedirs(save_dir, exist_ok=True)

    # Presenter / Role 列を追加
    df = df_scores.copy()
    df["Presenter"] = df["Expert"].map(lambda k: presenter_role_dict.get(k, {}).get("Presenter", k))
    df["Role"] = df["Expert"].map(lambda k: presenter_role_dict.get(k, {}).get("Role", "unknown").lower())
    print(df)
    print(df["Role"])

    # Roleごとの色
    role_colors = {
        "academic": "#1f77b4",
        "citizen": "#2ca02c",
        "private": "#ff7f0e",
        "public": "#d62728",
        "unknown": "gray"
    }
    df["Color"] = df["Role"].map(lambda r: role_colors.get(r, "gray"))

    # スコア順にソート
    df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)

    # プロット設定
    plt.figure(figsize=(8, max(4, len(df_sorted) * 0.4)))
    sns.barplot(
        data=df_sorted,
        y="Presenter",
        x="Score",
        hue="Role",
        dodge=False,
        palette=role_colors
    )

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("左派 ←→ 右派 方向の射影スコア（負: 左派寄り / 正: 右派寄り）")
    plt.ylabel("情報提供者（Presenter）")
    plt.title("情報提供者ごとの政治的傾向スコア")
    plt.legend(title="Role", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # 保存
    save_path = os.path.join(save_dir, f"political_projection_scores_{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 射影スコア一覧を出力しました: {save_path}")

def plot_political_scatter(df_plot, presenter_role_dict, save_dir, file_name=None):
    os.makedirs(save_dir, exist_ok=True)

    df = df_plot.copy()
    df["Presenter"] = df["Expert"].map(lambda k: presenter_role_dict.get(k, {}).get("Presenter", k))
    df["Role"] = df["Expert"].map(lambda k: presenter_role_dict.get(k, {}).get("Role", "unknown")).astype(str).str.lower()

    role_colors = {
        "academic": "#1f77b4",
        "citizen": "#2ca02c",
        "private": "#ff7f0e",
        "public": "#d62728",
        "unknown": "gray"
    }

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="ScoreRel",
        y="RefCount",
        hue="Role",
        palette=role_colors
    )

    # 点数が多くなったらラベル過密になるので、必要なら条件分岐で制限してもOK
    for _, r in df.iterrows():
        plt.text(r["ScoreRel"], r["RefCount"], str(r["Presenter"]),
                 fontsize=9, ha="left", va="bottom")

    plt.xlabel("政治的傾向（相対スコア：平均との差）")
    plt.ylabel("参照回数（Top1として参照された回数）")
    plt.title("情報提供者の相対的政治傾向 × 参照回数")
    plt.legend(title="Role", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"political_scatter_{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ scatter saved: {save_path}")

# def visualize_umap_from_average_vectors_with_meta(
#     average_vector_for_each_expert: dict,
#     meta_by_key: dict,                         # 例: {"lecture2_3": {"Presenter":"厚木市2", "Role":"public", ...}, ...}
#     save_dir=".",
#     n_neighbors=10,
#     min_dist=0.1,
#     reference_data=None,
#     model_name="cl-tohoku/bert-base-japanese-v3"
# ):
#     # 役割ごとの色
#     ROLE_COLORS = {
#         "academic": "#1f77b4",
#         "citizen":  "#2ca02c",
#         "private":  "#ff7f0e",
#         "public":   "#d62728",
#     }
#     os.makedirs(save_dir, exist_ok=True)

#     # 元データ
#     keys   = list(average_vector_for_each_expert.keys())
#     vecs   = [_to_np32(average_vector_for_each_expert[k]).reshape(-1) for k in keys]

#     # 参考（基準）を末尾に追加
#     if reference_data is not None:
#         refs = compute_reference_vectors(reference_data, model_name=model_name)
#         for name, v in refs.items():
#             keys.append(name)
#             vecs.append(_to_np32(v).reshape(-1))

#     # 次元合わせ（最頻次元のみ残す）
#     dims = [v.shape[0] for v in vecs]
#     if len(set(dims)) > 1:
#         from collections import Counter
#         common = Counter(dims).most_common(1)[0][0]
#         keep = [i for i, d in enumerate(dims) if d == common]
#         keys = [keys[i] for i in keep]
#         vecs = [vecs[i] for i in keep]

#     # 非有限除外
#     keep = [i for i, v in enumerate(vecs) if _finite(v)]
#     keys = [keys[i] for i in keep]
#     vecs = [vecs[i] for i in keep]

#     if not keys:
#         raise ValueError("UMAP: 可視化可能なベクトルがありません。")

#     X = np.stack(vecs)
#     n = X.shape[0]

#     # 全点ほぼ同一なら円形に微小配置
#     if np.allclose(X, X[0], atol=1e-7):
#         t = np.linspace(0, 2*np.pi, n, endpoint=False)
#         coords = np.c_[1e-3*np.cos(t), 1e-3*np.sin(t)]
#     else:
#         if n == 1:
#             coords = np.array([[0.0, 0.0]], dtype=np.float32)
#         elif n == 2:
#             coords = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
#         else:
#             nn = min(n_neighbors, n - 1)    # 必ず n_neighbors < n_samples
#             reducer = umap.UMAP(n_neighbors=nn, min_dist=min_dist, metric="cosine", random_state=42)
#             coords = reducer.fit_transform(X)

#     # 可視化用DF（表示名と色をここで作る）
#     rows = []
#     for key, (x, y) in zip(keys, coords):
#         is_ref = key.startswith("REF_")
#         if is_ref:
#             disp = "Reference RIGHT" if key == "REF_RIGHT" else "Reference LEFT"
#             role = "reference"
#             color = "#d62728" if key == "REF_RIGHT" else "#1f77b4"
#             marker = "*" if key == "REF_RIGHT" else "X"
#         else:
#             meta = meta_by_key.get(key, {})
#             presenter = meta.get("Presenter", key)
#             role = str(meta.get("Role", "unknown")).lower()
#             disp = presenter
#             color = ROLE_COLORS.get(role, "gray")
#             marker = "o"
#         rows.append({"Key": key, "Display": disp, "Role": role, "x": float(x), "y": float(y),
#                      "Color": color, "Marker": marker})
#     df_vis = pd.DataFrame(rows)

#     # 描画
#     plt.figure(figsize=(10, 6))
#     # 専門家（role != reference）
#     for role, group in df_vis[df_vis["Role"] != "reference"].groupby("Role"):
#         plt.scatter(group["x"], group["y"], s=80, alpha=0.85,
#                     color=ROLE_COLORS.get(role, "gray"), label=role)
#         for _, r in group.iterrows():
#             plt.text(r["x"], r["y"], r["Display"], fontsize=8, color="black", ha="left", va="center")

#     # 基準（強調）
#     ref = df_vis[df_vis["Role"] == "reference"]
#     for _, r in ref.iterrows():
#         plt.scatter([r["x"]], [r["y"]], s=160, c=r["Color"], marker=r["Marker"],
#                     edgecolors="k", linewidths=0.8,
#                     label=r["Display"])
#         plt.text(r["x"], r["y"], r["Display"], fontsize=10, fontweight="bold", ha="left")

#     plt.title("情報提供者の発言内容と、右派・左派それぞれの代表的言説との意味的類似度の分析")
#     plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
#     plt.legend(loc="best")
#     plt.tight_layout()

#     # 保存
#     plot_path = os.path.join(save_dir, "umap_ideological_embeddings.png")
#     csv_path  = os.path.join(save_dir, "umap_ideological_embeddings.csv")
#     plt.savefig(plot_path, dpi=300); plt.close()
#     df_vis.to_csv(csv_path, index=False)
#     print(f"✅ Saved UMAP figure to: {plot_path}")
#     print(f"✅ Saved CSV to: {csv_path}")
#     return df_vis

# def topic_analysis_from_pickles(input_materials_pkl_pth_list):
#     """
#     各pickleに含まれる専門家の文章を集約してトピック抽出（BERTopic）
#     - 空/短文の除外
#     """
#     all_texts = []
#     for pkl_path in tqdm(input_materials_pkl_pth_list, desc="Loading texts for BERTopic"):
#         texts = input_organize(pickle_load(pkl_path))
#         # 短すぎる・空は除外（BERTopicが落ちにくくなる）
#         texts = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) > 0]
#         all_texts.extend(texts)

#     if len(all_texts) == 0:
#         raise ValueError("BERTopic: 入力テキストが空です（抽出できません）。")

#     topic_model = BERTopic(language="japanese")
#     topics, probs = topic_model.fit_transform(all_texts)

#     print("🧭 BERTopic トピック上位:")
#     print(topic_model.get_topic_info().head())
#     return topic_model
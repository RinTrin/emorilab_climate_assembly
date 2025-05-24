import pickle
import os
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
from transformers import pipeline as transformers_pipeline
from collections import defaultdict

# GiNZAモデルで日本語文分割
nlp = spacy.load("ja_ginza")

def pickle_load(pth):
    with open(pth, 'rb') as f:
        return pickle.load(f)

def split_into_sentences(text_list):
    sentences = []
    for text in text_list:
        doc = nlp(text)
        sentences.extend([sent.text.strip() for sent in doc.sents])
    return sentences

def input_organize(input_data):
    new_input_data = []
    for input_text in input_data:
        if len(input_text) <= 10:
            continue
        elif input_text.isdigit() or input_text.isnumeric():
            continue
        else:
            new_input_data.append(input_text)
    return new_input_data

def analyze_simmality(action_plans_pkl_pth_list, input_materials_pkl_pth_list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    # 全SourceFileから文と出典ファイル名を収集
    all_input_sentences = []
    all_input_sources = []

    for input_pkl_pth in input_materials_pkl_pth_list:
        input_filename = os.path.basename(input_pkl_pth)
        input_data = pickle_load(input_pkl_pth)
        input_data_organized = input_organize(input_data)
        input_sentences = split_into_sentences(input_data_organized)

        all_input_sentences.extend(input_sentences)
        all_input_sources.extend([input_filename] * len(input_sentences))

    # 一括エンコード
    all_input_embeddings = model.encode(all_input_sentences, convert_to_tensor=True)

    for action_pkl_pth in action_plans_pkl_pth_list:
        action_data = pickle_load(action_pkl_pth)
        action_data_organized = input_organize(action_data)
        action_sentences = split_into_sentences(action_data_organized)
        action_embeddings = model.encode(action_sentences, convert_to_tensor=True)

        for i, action_sentence in enumerate(action_sentences):
            cosine_scores = util.cos_sim(action_embeddings[i], all_input_embeddings)[0]
            topk = torch.topk(cosine_scores, k=3)
            top_scores = topk.values.tolist()
            top_indices = topk.indices.tolist()

            result_row = {
                "ActionPlan": action_sentence
            }

            for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
                result_row[f"Top{rank}_Score"] = f"{score:.05}"
                result_row[f"Top{rank}_Text"] = all_input_sentences[idx]
                result_row[f"Top{rank}_SourceFile"] = all_input_sources[idx]

            results.append(result_row)

    df = pd.DataFrame(results)

    # 保存
    save_dir = os.path.join(Path(__file__).parents[1], "analysis_results/each_sentence_all_files")
    os.makedirs(save_dir, exist_ok=True)
    csv_pth = os.path.join(save_dir, "analyzed_top3_across_all_files.csv")
    df.to_csv(csv_pth, index=False, encoding="utf-8-sig")

    return csv_pth




def analyze_political_leaning(csv_path, presenter_role_dict, save_dir="."):
    """
    日本語文に基づいて、Presenter/Roleごとの文章の政治的傾向（右派/左派）を分析。
    Zero-shot分類器を用いて、日本語文で右派・左派ラベルとの類似性スコアを測定。

    Returns:
        df_out: 結果DataFrame
        csv_path: 保存先CSVパス
    """
    
    df = pd.read_csv(csv_path)
    # Presenter/Role列をマージ
    df["Presenter"] = df["Top1_SourceFile"].map(lambda x: presenter_role_dict.get(x, {}).get("Presenter", "Unknown"))
    df["Role"] = df["Top1_SourceFile"].map(lambda x: presenter_role_dict.get(x, {}).get("Role", "Unknown"))


    # 日本語をそのまま扱える zero-shot pipeline（日本語拡張可）
    classifier = transformers_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    results = []
    grouped = df.groupby(['Presenter', 'Role'])

    for (presenter, role), group_df in grouped:
        texts = group_df['Top1_Text'].dropna().tolist()

        left_scores = []
        right_scores = []

        for text in texts:
            try:
                result = classifier(
                    sequences=text,
                    candidate_labels=["左派", "右派"],
                    hypothesis_template="この文章は{}の立場から書かれている。"
                )

                scores = dict(zip(result["labels"], result["scores"]))
                left_scores.append(scores.get("左派", 0))
                right_scores.append(scores.get("右派", 0))
            except Exception as e:
                print(f"[警告] 分類失敗: {text[:30]}... → {e}")

        avg_left = sum(left_scores) / len(left_scores) if left_scores else 0
        avg_right = sum(right_scores) / len(right_scores) if right_scores else 0

        leaning = "左派" if avg_left > avg_right else "右派" if avg_right > avg_left else "中立"

        results.append({
            "Presenter": presenter,
            "Role": role,
            "Avg_Left_Score": round(avg_left, 3),
            "Avg_Right_Score": round(avg_right, 3),
            "Leaning": leaning,
            "SampleCount": len(texts)
        })

    df_out = pd.DataFrame(results)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "presenter_political_leaning.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return df_out, csv_path
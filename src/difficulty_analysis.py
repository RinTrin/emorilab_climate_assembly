import os
import re
import MeCab
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer

# MeCabの設定
mecab = MeCab.Tagger("-Owakati -r /opt/homebrew/etc/mecabrc")

# テキストデータの読み込み
def load_text_files(directory="/Users/rintrin/codes/emorilab_climate_assembly/db_txt"):  # テキストファイルのフォルダ
    texts = []
    filenames = []
    for filepath in glob.glob(os.path.join(directory, "*/*/*.txt")):
        with open(filepath, "r", encoding="utf-8") as file:
            texts.append(file.read())
            filenames.append("/".join(filepath.split(os.sep)[-3:]))
    return filenames, texts

# 漢字のカウント
def count_kanji(text):
    return len(re.findall(r"[一-龥]", text))

# 文章のトークナイズ
def tokenize_text(text):
    parsed_text = mecab.parse(text)
    return parsed_text.strip().split() if parsed_text else []

# 読みやすさ指標の計算
def calculate_readability(text):
    words = tokenize_text(text)
    total_words = len(words)
    total_chars = len(text)
    kanji_count = count_kanji(text)
    
    # 難解語のカウント（4文字以上の漢語）
    difficult_words = [w for w in words if len(w) >= 4 and re.search(r"[一-龥]", w)]
    difficult_word_ratio = len(difficult_words) / total_words if total_words > 0 else 0
    
    # 漢字比率
    kanji_ratio = kanji_count / total_chars if total_chars > 0 else 0
    
    # 平均単語長
    avg_word_length = sum(len(w) for w in words) / total_words if total_words > 0 else 0
    
    return {
        "難解語比率": round(difficult_word_ratio, 4),
        "漢字比率": round(kanji_ratio, 4),
        "平均単語長": round(avg_word_length, 4)
    }

# TF-IDFの計算と専門用語抽出
def extract_specialized_terms(texts, top_n=50):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 各単語の平均TF-IDFスコアを計算
    avg_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(avg_tfidf_scores)[::-1]  # 降順にソート

    # 上位N単語を専門用語候補とする
    specialized_terms = set(feature_names[sorted_indices[:top_n]])
    return specialized_terms

# TF-IDFによる専門用語割合の計算
def calculate_tfidf_specialized_ratio(texts, specialized_terms):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 専門用語のTF-IDFスコアを取得
    specialized_scores = []
    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = dict(zip(feature_names, tfidf_matrix[i].toarray().flatten()))
        specialized_tfidf_sum = sum(tfidf_scores.get(word, 0) for word in specialized_terms)
        total_tfidf_sum = sum(tfidf_scores.values())
        specialized_ratio = specialized_tfidf_sum / total_tfidf_sum if total_tfidf_sum > 0 else 0
        specialized_scores.append(round(specialized_ratio, 4))
    
    return specialized_scores

# メイン処理
filenames, texts = load_text_files()

# 専門用語を自動抽出
specialized_terms = extract_specialized_terms(texts, top_n=50)

# 各テキストの読みやすさ指標を計算
readability_scores = [calculate_readability(text) for text in texts]

# TF-IDF専門用語割合を計算
tfidf_specialized_ratios = calculate_tfidf_specialized_ratio(texts, specialized_terms)

# データフレームに整理
df = pd.DataFrame(readability_scores)
df["TF-IDF専門用語割合"] = tfidf_specialized_ratios
df["ファイル名"] = filenames

# 都市名の抽出
df["都市名"] = df["ファイル名"].apply(lambda x: x.split("/")[0])

# 数値カラムのみを選択して変換
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].round(4)

# グラフの作成
def plot_metrics(df):
    metrics = ["難解語比率", "漢字比率", "平均単語長", "TF-IDF専門用語割合"]
    cities = df["都市名"].unique()
    rows, cols = 3, 3  # 3行3列のグリッドレイアウト
    
    for metric in metrics:
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
        fig.suptitle(f"{metric} の都市別プロット")
        
        for ax, city in zip(axes.flatten(), cities):
            city_df = df[df["都市名"] == city]
            ax.scatter(city_df["ファイル名"].apply(lambda x: "inputmaterial" if "inputmaterial" in x else "actionplan"), city_df[metric], alpha=0.7)
            ax.set_title(city)
            ax.grid()
            if metric == "難解語比率":
                ax.set_ylim(0, 0.03)
            elif metric == "漢字比率":
                ax.set_ylim(0.1, 0.5)
            elif metric == "平均単語長":
                ax.set_ylim(1.5, 2.4)
            elif metric == "TF-IDF専門用語割合":
                ax.set_ylim(0.1, 0.4)
        
        plt.xlabel("カテゴリ")
        plt.tight_layout()
        plt.savefig(f"/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/imgs/{metric}.png")
        plt.show()

plot_metrics(df)

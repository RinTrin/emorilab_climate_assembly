# 置き換え対象：上部のGiNZA関連は削除またはコメントアウトしてOK
import pickle
import os
import gc
import pandas as pd
from pathlib import Path
import torch
from torch.nn.functional import normalize
from transformers import BertJapaneseTokenizer, BertModel
from utils import pickle_load, input_organize


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=32):
        embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = self.tokenizer.batch_encode_plus(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            out = self.model(**encoded)
            sent_emb = self._mean_pooling(out, encoded["attention_mask"]).detach().cpu()
            embs.append(sent_emb)
            # 早めに解放
            del encoded, out, sent_emb
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(embs, dim=0) if len(embs) > 0 else torch.empty(0, self.model.config.hidden_size)


def calc_similarity_ja(action_plans_pkl_pth_list, input_materials_pkl_pth_list):
    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    model = SentenceBertJapanese(MODEL_NAME)

    # ===== 1) 入力素材（inputmaterial）を一度だけまとめてエンコード =====
    all_input_sentences = []
    all_input_sources = []
    for input_pkl in input_materials_pkl_pth_list:
        src = os.path.basename(input_pkl)
        sents = input_organize(pickle_load(input_pkl))
        all_input_sentences.extend(sents)
        all_input_sources.extend([src] * len(sents))

    with torch.no_grad():
        input_emb = model.encode(all_input_sentences, batch_size=64)  # まとめて
        # L2正規化 + float16でメモリ半減
        input_emb = normalize(input_emb, p=2, dim=1).to(dtype=torch.float16)

    results = []
    hidden = input_emb.shape[1]

    # ===== 2) アクション文をバッチで処理し、行列積で一括cos類似 =====
    for action_pkl in action_plans_pkl_pth_list:
        action_sents = input_organize(pickle_load(action_pkl))

        # バッチ処理
        for i in range(0, len(action_sents), 64):
            batch_sents = action_sents[i:i+64]

            with torch.no_grad():
                act_emb = model.encode(batch_sents, batch_size=64)
                act_emb = normalize(act_emb, p=2, dim=1).to(dtype=torch.float16)

                # cos類似 = 正規化後の内積
                # [B, H] x [H, N] -> [B, N]
                sims = (act_emb.to(dtype=torch.float32) @ input_emb.to(dtype=torch.float32).T)

                # 各行でTop-3
                top_scores, top_idx = torch.topk(sims, k=3, dim=1)

            # 結果へ
            for row_idx, action_sentence in enumerate(batch_sents):
                scores = top_scores[row_idx].tolist()
                idxs = top_idx[row_idx].tolist()

                row = {"ActionPlan": action_sentence}
                for rank, (sc, j) in enumerate(zip(scores, idxs), start=1):
                    row[f"Top{rank}_Score"] = f"{sc:.05f}"
                    row[f"Top{rank}_Text"] = all_input_sentences[j]
                    row[f"Top{rank}_SourceFile"] = all_input_sources[j]
                results.append(row)

            # 解放
            del act_emb, sims, top_scores, top_idx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ===== 3) 保存 =====
    df = pd.DataFrame(results)
    save_dir = os.path.join(Path(__file__).parents[1], "analysis_results/each_sentence_all_files")
    os.makedirs(save_dir, exist_ok=True)
    csv_pth = os.path.join(save_dir, "analyzed_top3_across_all_files_ja.csv")
    df.to_csv(csv_pth, index=False, encoding="utf-8-sig")
    # メモリ解放
    del input_emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("FIN CALC SIMILARITY JA")
    return csv_pth
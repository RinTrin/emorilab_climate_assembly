import os
import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import AutoTokenizer, BertJapaneseTokenizer, BertModel

# ==== 定数 ====
LABELS = ["O", "BR"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 512          # BERTの上限（CLS/SEP含む）
STRIDE_WORDS = 50          # チャンク間オーバーラップ（語単位）

# ==== モデル定義 ====
class BertCRF(nn.Module):
    def __init__(self, tagset_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())

# ==== tokenizer を用意（Fast優先、無ければSlowでフォールバック） ====
def get_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese", use_fast=True)
        if not getattr(tok, "is_fast", False):
            raise ValueError("fast tokenizer not available")
        return tok, True
    except Exception:
        tok = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        return tok, False

# ==== 入力ファイルから単語列だけ抽出 ====
def load_sentences(input_path):
    """
    入力ファイルを文単位の「語（=トークン）列」のリストに変換して返す。
    - 空行があれば文区切りとして扱う（複数行で1文を作るのにも対応）
    - 行が既に分かち書き（空白/タブ区切り）の場合は split() で全部採用
    - 生テキスト（日本語で空白が少ない）なら文字単位に分割して全量採用
    """
    sentences = []
    buffer = []  # 文をまたぐ場合に行を溜める

    def flush_buffer():
        nonlocal buffer, sentences
        if not buffer:
            return
        # バッファを結合して1本のテキストに
        text = "".join(buffer)
        # もし空白/タブが十分に含まれていれば分かち書きとみなす
        has_whitespace = (" " in text) or ("\t" in text)
        if has_whitespace:
            tokens = text.split()  # 全トークンを採用（先頭だけにしない）
        else:
            tokens = list(text)    # 生テキストは文字単位で全量採用
        if tokens:
            sentences.append(tokens)
        buffer = []

    with open(input_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "":
                # 空行で文を確定
                flush_buffer()
            else:
                buffer.append(line.strip())
        # ファイル末尾に文が残っていれば確定
        flush_buffer()

    return sentences


# ==== 出力ファイル名 ====
def generate_output_path(input_path):
    base, ext = os.path.splitext(input_path)
    return f"{base}_punc_added{ext}"

# ==== Slowトークナイザ用: 擬似 word_ids() を生成 ====
def build_word_ids_slow(words, tokenizer):
    """
    CLS=先頭None, 各語のサブワード数だけ語インデックスを並べ、末尾にSEP=None を付与。
    """
    wids = [None]  # CLS
    for i, w in enumerate(words):
        # subword 分割個数を取得（[UNK]でも1つ返る）
        num_sub = len(tokenizer.tokenize(w))
        if num_sub <= 0:
            num_sub = 1
        wids.extend([i] * num_sub)
    wids.append(None)  # SEP
    return wids

# ==== 語列をトークン上限に収まるようチャンク（word単位） ====
def chunk_words_by_token_budget(words, tokenizer, is_fast, max_seq_len=MAX_SEQ_LEN, stride_words=STRIDE_WORDS):
    spans = []
    n = len(words)
    start = 0
    while start < n:
        end = start + 1
        best_end = end
        while end <= n:
            enc = tokenizer(
                words[start:end],
                is_split_into_words=True,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            seq_len = enc["input_ids"].shape[1]  # CLS/SEP含む
            if seq_len <= max_seq_len:
                best_end = end
                end += 1
            else:
                break
        spans.append((start, best_end))
        if best_end >= n:
            break
        start = max(best_end - stride_words, start + 1)
    return spans

# ==== 1文（words）に対するラベル推論（語ラベル列を返す） ====
def predict_labels_for_words(words, tokenizer, model, is_fast):
    if len(words) == 0:
        return []

    spans = chunk_words_by_token_budget(words, tokenizer, is_fast, MAX_SEQ_LEN, STRIDE_WORDS)
    word_labels = [None] * len(words)

    model.eval()
    with torch.no_grad():
        for (s, e) in spans:
            sub_words = words[s:e]
            enc = tokenizer(
                sub_words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            token_labels = preds[0]  # 1文想定

            # word_ids の取得（Fastなら enc.word_ids()、Slowなら自作）
            if is_fast:
                wids = enc.word_ids()
            else:
                wids = build_word_ids_slow(sub_words, tokenizer)

            # サブワード→語：各語の先頭サブワードのラベルを採用（重複は前スパン優先）
            seen_local = set()
            for tok_idx, wid in enumerate(wids):
                if wid is None:
                    continue
                if wid not in seen_local:
                    seen_local.add(wid)
                    global_wid = s + wid
                    if word_labels[global_wid] is None:
                        lab = token_labels[tok_idx] if tok_idx < len(token_labels) else 0
                        word_labels[global_wid] = int(lab)

    for i in range(len(word_labels)):
        if word_labels[i] is None:
            word_labels[i] = 0  # "O"
    return word_labels

# ==== 本処理：句読点推論・保存 ====
def predict_and_insert_punctuation(input_path, model_path="punctuation_train/bcrf_model.pt"):
    tokenizer, is_fast = get_tokenizer()
    model = BertCRF(tagset_size=len(LABELS)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    sentences = load_sentences(input_path)
    output_path = generate_output_path(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for words in sentences:
            pred_labels = predict_labels_for_words(words, tokenizer, model, is_fast)

            sentence = ""
            for word, label_id in zip(words, pred_labels):
                sentence += word
                if id2label[label_id] == "BR":
                    sentence += "。"
            if not sentence.endswith("。"):
                sentence += "。"
            f.write(sentence + "\n")

    print(f"✅ 推論完了：{output_path}")

# ==== 実行部 ====
if __name__ == "__main__":
    predict_and_insert_punctuation("test.txt")

import pandas as pd
import os
import numpy as np
import yaml
import subprocess
import io
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from openai import OpenAI  # ← これが正しいv1.xの使い方

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"

def filter_top1_score(csv_path, threshold=0.7):
    """
    Top1_Score列の値が指定されたしきい値以下の行を削除し、
    新しいCSVを作成してそのパスを返す関数。

    Parameters:
    csv_path (str): 入力CSVファイルのパス
    threshold (float): Top1_Scoreのしきい値（この値以下を削除）

    Returns:
    str: 新しく作成されたCSVファイルのパス（analyzed_csv_pth）
    """
    # CSV読み込み
    df = pd.read_csv(csv_path)

    # フィルタリング
    df_filtered = df[df['Top1_Score'] > threshold]

    # 新しいファイル名を生成
    base, ext = os.path.splitext(csv_path)
    analyzed_csv_pth = f"{base}_filtered{ext}"

    # 新しいCSVとして保存
    df_filtered.to_csv(analyzed_csv_pth, index=False)

    return analyzed_csv_pth


# ✅ Whisper API用のOpenAIクライアント（APIキー埋め込み or 環境変数）
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# または直接書く場合：
# client = OpenAI(api_key="sk-...")

# === Step 1: YouTube音声をWAVとして取得 ===
def get_audio_segment_from_youtube(url: str) -> AudioSegment:
    yt_proc = subprocess.Popen(
        ["yt-dlp", "-f", "bestaudio", "-o", "-", url],
        stdout=subprocess.PIPE
    )
    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"],
        stdin=yt_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    audio_bytes = ffmpeg_proc.stdout.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    return audio

# === Step 2: 音声分割 ===
def split_audio(audio: AudioSegment, chunk_length_ms: int = 10 * 60 * 1000):
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# === Step 3: Whisper APIで書き起こし（v1.xスタイル） ===
def transcribe_chunk_with_whisper(chunk: AudioSegment, language='ja', initial_prompt=None):
    with NamedTemporaryFile(suffix=".wav") as temp_file:
        chunk.export(temp_file.name, format="wav")
        with open(temp_file.name, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="text"
            )
    return response

# === Step 4: 実行制御 ===
def transcribe_youtube_to_text(url: str, output_path="transcript.txt", language='ja'):
    print("🔊 音声をYouTubeから取得中...")
    audio = get_audio_segment_from_youtube(url)

    print(f"✂️ 音声長: {len(audio) / 60000:.2f} 分 → 分割処理中...")
    chunks = split_audio(audio)
    print(f"🔹 チャンク数: {len(chunks)}")

    transcript_parts = []
    for i, chunk in enumerate(chunks):
        print(f"🧠 Whisper API実行中... チャンク {i+1}/{len(chunks)}")
        try:
            text = transcribe_chunk_with_whisper(
                chunk,
                language=language,
                initial_prompt=None
            )
            transcript_parts.append(text.strip())
        except Exception as e:
            print(f"⚠️ チャンク {i+1} の処理中にエラーが発生しました: {e}")
            transcript_parts.append(f"[Chunk {i+1}]\n[エラー発生]\n")

    full_transcript = "\n".join(transcript_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)

    print(f"✅ 書き起こし完了: {output_path}")

def merge_materialtext_and_youtubetext(city_name, ROOT):
    import yaml
    from pathlib import Path

    # city_name は既に定義済み
    # city_name = "Atugi"

    root = Path("/Users/rintrin/codes/emorilab_climate_assembly")

    # 出力ディレクトリ
    out_dir = root / "db_merged_txt" / city_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 見出し
    HEADER_ORG = "=== Original Input Material ==="
    HEADER_YT  = "=== YouTube Transcription (punc_added) ==="

    # YAML 読み込み
    with open(root / "src_rq2/inputmaterial_info.yaml", "r", encoding="utf-8") as f:
        lecture_info = yaml.safe_load(f)
    lecture_info_city = lecture_info[str(city_name)]

    processed, missing_files = [], []

    for lecture_name, lecture_each_info in lecture_info_city.items():
        # 入力資料パス
        input_path = root / "db_txt" / city_name / "inputmaterial" / lecture_each_info["Inputmaterial_txt_path"]

        # YouTube 側パス（punc_added固定）
        yt_rel = lecture_each_info["Youtube_txt_path"]
        yt_stem = Path(yt_rel).stem.replace("_youtube_txt", "_youtube_txt_punc_added")
        yt_path = root / "db_youtube_txt" / city_name / f"{yt_stem}.txt"

        # 存在チェック
        if not input_path.exists():
            missing_files.append((lecture_name, "inputmaterial", input_path))
            continue
        if not yt_path.exists():
            missing_files.append((lecture_name, "youtube_punc_added", yt_path))
            continue

        # 出力パス
        base_name = Path(input_path).stem
        out_path = out_dir / f"{base_name}_merged.txt"

        # 読み込み
        with open(input_path, "r", encoding="utf-8") as f:
            input_text = f.read().strip()
        with open(yt_path, "r", encoding="utf-8") as f:
            yt_text = f.read().strip()

        # マージ
        merged = (
            f"{HEADER_ORG}\n{input_text}\n\n"
            f"{HEADER_YT}\n{yt_text}\n"
        )

        # 書き出し
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(merged)

        processed.append((lecture_name, out_path.name))

    # 結果出力
    print(f"\n=== {city_name} のマージ結果 ===")
    if processed:
        print("✅ マージ完了:")
        for lec, outn in processed:
            print(f" - {lec} → {outn}")
    if missing_files:
        print("\n⚠️ ファイル未検出:")
        for lecture, kind, path in missing_files:
            print(f" - {lecture} [{kind}]: {path}")
    else:
        print("すべてのテキストをマージしました ✅")

import pickle
def pickle_load(pth):
    with open(pth, 'rb') as f:
        return pickle.load(f)

def input_organize(input_data):
    # すでに文単位のリスト前提。念のためフィルタのみ。
    new_input_data = []
    for s in input_data:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if len(s) <= 10:
            continue
        if s.isdigit() or s.isnumeric():
            continue
        new_input_data.append(s)
    return new_input_data

def _to_np32(x):
    if isinstance(x, list):
        x = np.array(x)
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _finite(a): return np.isfinite(a).all()


import os
import json
import pickle
import hashlib
from datetime import datetime
from openai import OpenAI
from typing import Optional

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def generate_reference_data_climate_cached(
    n_texts_each: int = 5,
    cache_dir: str = "/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/political_analysis/reference_cache",
    model: str = "gpt-5.2",
    seed: Optional[int] = 42,
    force_refresh: bool = False,
) -> dict:
    """
    気候変動テーマの right/left 参照文を GPT-5.2 で生成し、
    json / pkl にキャッシュして再利用する。

    戻り値:
        {"right_texts": [...], "left_texts": [...]}
    """
    _ensure_dir(cache_dir)

    # --- 仕様（これが同じなら同じキャッシュを使う） ---
    spec = {
        "theme": "climate_change_ideology_axis_jp",
        "n_texts_each": n_texts_each,
        "length_chars": [120, 280],
        "sentences": [3, 7],
        "seed": seed,
        "definition_right": "規制強化や急進的削減に慎重。不確実性、自然変動、コスト、雇用、国益、エネルギー安全保障を重視し、適応や技術で対応する立場。",
        "definition_left": "気候危機（1.5℃など）を前提に、迅速な排出削減、再エネ拡大、規制、公正な移行や気候正義を重視する立場。",
    }

    spec_key = _sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True))[:16]
    json_path = os.path.join(cache_dir, f"reference_data_{spec_key}.json")
    pkl_path  = os.path.join(cache_dir, f"reference_data_{spec_key}.pkl")

    # --- キャッシュ優先 ---
    if not force_refresh:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

    # --- GPT生成 ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません")

    client = OpenAI(api_key=api_key)

    schema = {
        "name": "reference_data_schema",
        "schema": {
            "type": "object",
            "properties": {
                "right_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": n_texts_each,
                    "maxItems": n_texts_each,
                },
                "left_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": n_texts_each,
                    "maxItems": n_texts_each,
                },
            },
            "required": ["right_texts", "left_texts"],
            "additionalProperties": False,
        },
    }

    prompt = f"""
    テーマ：気候変動（温暖化・脱炭素・エネルギー政策）

    条件：
    - 日本語
    - 政治家の答弁・演説風
    - 各文 3〜7文、120〜280文字程度
    - right_texts を {n_texts_each} 本
    - left_texts を {n_texts_each} 本
    - 実在の人物名・政党名・メディア名・URLは禁止

    立場定義：
    - right_texts：
    {spec["definition_right"]}

    - left_texts：
    {spec["definition_left"]}

    必ず指定スキーマに一致する JSON のみを出力してください。
    """

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "あなたは日本語の気候変動専門家言説データ生成アシスタントです。"},
            {"role": "user", "content": prompt.strip()},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "reference_data",
                "strict": True,
                "schema": schema["schema"],  # ← あなたが作った中身を使う
            }
        }
    )

    data = json.loads(resp.output_text)

    # --- 軽い検証 ---
    for k in ("right_texts", "left_texts"):
        if k not in data or len(data[k]) != n_texts_each:
            raise ValueError(f"生成結果が不正です: {k}")

    # --- 保存 ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data



def add_presenter_columns_to_analyzed_csv(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname):
    """
    analyzed_csv_pth に含まれる
    Top1/2/3_SourceFile から Presenter 名を引き当て、
    TopX_Presenter 列を追加して同じCSVに上書き保存する
    """
    df = pd.read_csv(analyzed_csv_pth)

    for i in [1, 2, 3]:
        src_col = f"Top{i}_SourceFile"
        pres_col = f"Top{i}_Presenter"

        if src_col not in df.columns:
            continue

        def resolve_presenter(src):
            if pd.isna(src):
                return None
            key = str(src).replace("_youtube_txt_segmented.pkl", "")
            return presenter_role_dict.get(key, {}).get("Presenter", "UNKNOWN")

        df[pres_col] = df[src_col].apply(resolve_presenter)

        # SourceFile の直後に Presenter 列を移動
        cols = list(df.columns)
        cols.remove(pres_col)
        insert_idx = cols.index(src_col) + 1
        cols.insert(insert_idx, pres_col)
        df = df[cols]

    df.to_csv(f"/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files/check_{actionplan_excel_sheetname}.csv", index=False)
    print(f"✅ Presenter列を追加して上書き保存しました: {analyzed_csv_pth}")


def return_presenter_role_dict(city_name, measure_youtube_length=False):
    YAML_PATH = f"{ROOT}/src_rq2/inputmaterial_info.yaml"
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if measure_youtube_length:
        with open(YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for city, lectures in data.items():
            for lec, info in lectures.items():
                url = info.get("Youtube_path")
                if not url: continue
                try:
                    j = subprocess.run(
                        ["yt-dlp", "-j", "--no-playlist", url],
                        capture_output=True, text=True, check=True
                    )
                    dur = int(json.loads(j.stdout)["duration"])
                    info["PresentationLengthSecond"] = dur
                    print(f"{lec}: {dur}s")
                except Exception as e:
                    print(f"{lec}: error ({e})")

        with open(YAML_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    else:
        pass

    if city_name not in data:
        print("No City Name")
        raise KeyError

    return data[city_name]



# === 使用例 ===
if __name__ == "__main__":
    YOUTUBE_URL = "https://www.youtube.com/watch?v=E4oMkCvo62w"  # ← 動画URLをここに入れる
    transcribe_youtube_to_text(YOUTUBE_URL, output_path="4b-1.txt", language='ja')
#     import yaml

#     # YAMLファイルの読み込み
#     with open("/Users/rintrin/codes/emorilab_climate_assembly/src_rq2/inputmaterial_info.yaml", "r", encoding="utf-8") as f:
#         data = yaml.safe_load(f)

#     # テキストファイルを生成
#     for area, lectures in data.items():
#         for lecture_key, lecture_info in lectures.items():
#             txt_path = lecture_info.get("Youtube_txt_path")
#             txt_path = os.path.join("/Users/rintrin/codes/emorilab_climate_assembly/db_youtube_txt", txt_path)
#             if txt_path and not os.path.exists(txt_path):
#                 with open(txt_path, "w", encoding="utf-8") as txt_file:
#                     txt_file.write("")  # 空のファイルとして作成
#                 print(f"Created: {txt_path}")
#             else:
#                 print(f"Skipped (already exists or missing path): {txt_path}")

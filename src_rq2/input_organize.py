
import os
import re
import pickle
import yaml
import subprocess
from time import sleep
from pathlib import Path

from bunkai import Bunkai

from utils import transcribe_youtube_to_text, merge_materialtext_and_youtubetext
from punctuation_train.predict import predict_and_insert_punctuation

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"

def get_sentences(city_name, mode="actionplan"):
    """
    返り値: 文分割済みPKLファイルのパスのリスト
    - mode="inputmaterial": PDF→TXT, YouTube→TXT(+句読点)→ merge を実行し、
      **db_merged_txt/{city}/*.txt** を文分割して PKL を返す
    - mode="actionplan": PDF→TXT（マージなし）を文分割して PKL を返す
    
    内容
    都市ごとのPDFをTXTに変換する
    （inputmaterial の場合）YouTube音声を文字起こしし、句読点を挿入する
    （inputmaterial の場合）PDF由来TXTとYouTube由来TXTをマージする
    生成されたTXTを文分割（Bunkai）する
    文リストを *_segmented.pkl として保存し、そのパス一覧を返す
    """
    if city_name is None:
        raise ValueError("city_name must be provided")

    # 0) ベースフォルダ（PDF格納先）
    city_folder_pth = os.path.join(Path(__file__).parents[1], f'db_pdf/{city_name}')
    if mode == 'actionplan':
        city_folder_pth = os.path.join(city_folder_pth, 'actionplan')
    elif mode == 'inputmaterial':
        city_folder_pth = os.path.join(city_folder_pth, 'inputmaterial')

    # 1) PDF -> TXT（両モード共通）
    pdf_txt_files = []
    if os.path.isdir(city_folder_pth):
        for fname in os.listdir(city_folder_pth):
            if fname.endswith('.pdf'):
                pdf_pth = os.path.join(city_folder_pth, fname)
                out_txt = pdf_pth.replace("/db_pdf/", "/db_txt/").replace(".pdf", ".txt")
                os.makedirs(os.path.dirname(out_txt), exist_ok=True)
                if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                    pdf_txt_files.append(out_txt)
                else:
                    cmd = f"bash {ROOT}/src/pdf2text_mac_default.sh {pdf_pth} >> {out_txt}"
                    print(cmd)
                    subprocess.call(cmd, shell=True)
                    pdf_txt_files.append(out_txt)

    # 2) inputmaterial のみ: YouTube ASR (+句読点) と マージ
    target_txt_files = pdf_txt_files  # デフォはPDF由来TXT
    if mode == "inputmaterial":
        # ASR & 句読点
        with open(f'{ROOT}/src_rq2/inputmaterial_info.yaml', 'r', encoding='utf-8') as f:
            lecture_info = yaml.safe_load(f)
        lecture_info_city = lecture_info[str(city_name)]

        for lecture_name, lecture_each_info in lecture_info_city.items():
            YOUTUBE_URL = lecture_each_info.get("Youtube_path")
            yt_txt_name = lecture_each_info.get("Youtube_txt_path")
            if not yt_txt_name:
                continue

            OUTPUT_PATH = os.path.join(f"{ROOT}/db_youtube_txt/{city_name}", yt_txt_name)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            # 既存ASRの確認
            txt = ""
            if os.path.exists(OUTPUT_PATH):
                with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    txt = f.read()
            if txt.strip():
                print(f"{lecture_name} : IS ALREADY DONE")
            elif YOUTUBE_URL:
                transcribe_youtube_to_text(YOUTUBE_URL, output_path=OUTPUT_PATH, language='ja')

            # 句読点挿入（idempotent）
            PUNCTUATED_OUTPUT_PATH = OUTPUT_PATH.replace(".txt", "_punc_added.txt")
            if os.path.exists(PUNCTUATED_OUTPUT_PATH) and os.path.getsize(PUNCTUATED_OUTPUT_PATH) > 0:
                print(f"{PUNCTUATED_OUTPUT_PATH} : IS ALREADY DONE")
            else:
                predict_and_insert_punctuation(OUTPUT_PATH)

        # マージ実行（戻り値なしの実装想定）
        merge_materialtext_and_youtubetext(city_name, ROOT)

        # マージ後TXTを拾う（例：/db_merged_txt/{city}/*.txt）
        merged_dir = Path(f"{ROOT}/db_merged_txt/{city_name}")
        if merged_dir.exists():
            merged_txts = sorted(str(p) for p in merged_dir.glob("*.txt"))
            if merged_txts:
                target_txt_files = merged_txts  # 以降はマージ後TXTのみを使用

    # 3) 文分割 -> PKL（TXTと同ディレクトリに *_segmented.pkl を作成）
    segmented_pkl_pth_list = []
    bunkai = Bunkai()

    for txt_pth in target_txt_files:
        if not os.path.exists(txt_pth):
            print(f"[WARN] TXT not found: {txt_pth}")
            continue

        with open(txt_pth, 'r', encoding='utf-8') as f:
            text_data = f.read()

        # 文分割
        segmenter_list = list(bunkai(text_data))
        # 改行除去など軽い正規化
        segmenter_list = [s.replace('\n', '') for s in segmenter_list]

        # PKL 保存（TXTと同じ場所に *_segmented.pkl）
        segmented_pkl = txt_pth.replace(".txt", "_segmented.pkl")
        with open(segmented_pkl, 'wb') as f:
            pickle.dump(segmenter_list, f)

        segmented_pkl_pth_list.append(segmented_pkl)
        sleep(0.2)

    # ★ ここで PKL のリストを返す（calc_similarity_ja が想定）
    print(f"GET SENTENCE MODE:{mode} DONE")
    return segmented_pkl_pth_list


def get_sentences_annotated(city_name, mode="inputmaterial"):
    """
    返り値: 文分割済みPKLファイルのパスのリスト

    - mode="inputmaterial":
        db_youtube_txt_annotated/{city} 配下の .txt を読み、
        '.' 区切りで文リスト化して *_segmented.pkl を作成して返す
        （出力形式は従来の get_sentences と同じ）

    - mode="actionplan":
        今回は対象外（必要なら従来実装を残してそちらを呼んでね）
    """
    if city_name is None:
        raise ValueError("city_name must be provided")
    if mode not in ("actionplan", "inputmaterial"):
        raise ValueError("mode must be 'actionplan' or 'inputmaterial'")
    if mode != "inputmaterial":
        raise ValueError("This version is for mode='inputmaterial' only")

    # 0) 入力TXTフォルダ（.区切り文が大量にある場所）
    in_dir = Path(ROOT) / "db_youtube_txt_annotated" / str(city_name)
    if not in_dir.exists():
        raise FileNotFoundError(f"Folder not found: {in_dir}")

    # 1) 対象TXTを収集
    target_txt_files = sorted(str(p) for p in in_dir.glob("**/*.txt"))

    # 2) '.' 区切り -> PKL（TXTと同ディレクトリに *_segmented.pkl）
    segmented_pkl_pth_list = []
    split_pat = r"\.+"  # "...." の連続も区切り扱い

    for txt_pth in target_txt_files:
        if not os.path.exists(txt_pth):
            continue

        segmented_pkl = txt_pth.replace(".txt", "_segmented.pkl")
        if os.path.exists(segmented_pkl) and os.path.getsize(segmented_pkl) > 0:
            segmented_pkl_pth_list.append(segmented_pkl)
            continue

        with open(txt_pth, "r", encoding="utf-8", errors="ignore") as f:
            text_data = f.read()

        sentences = re.split(split_pat, text_data)
        sentences = [s.replace("\n", "").strip() for s in sentences]
        sentences = [s for s in sentences if s]  # 空文除去

        with open(segmented_pkl, "wb") as f:
            pickle.dump(sentences, f)

        segmented_pkl_pth_list.append(segmented_pkl)
        sleep(0.2)

    print(f"GET SENTENCE MODE:{mode} DONE")
    return segmented_pkl_pth_list
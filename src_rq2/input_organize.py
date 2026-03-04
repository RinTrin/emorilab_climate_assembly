
import os
import re
import pickle
import yaml
import subprocess
from time import sleep
from pathlib import Path
import pandas as pd

from bunkai import Bunkai

from utils import transcribe_youtube_to_text, merge_materialtext_and_youtubetext
from punctuation_train.predict import predict_and_insert_punctuation

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"

def get_sentences_actionplan_excel(excel_path, refresh_pickle=False, target_col="文章切り抜き", actionplan_excel_sheet_name=None):
    """
    excel_path: 1都市のactionplan Excel
    返り値: 生成した *_segmented.pkl のパス

    - 文分割しない
    - 各シートの target_col（デフォルト: 「文章切り抜き」）列だけを取得
    - Excelと同じディレクトリに {元xlsx名}_segmented.pkl を保存
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    segmented_pkl = excel_path.with_name(excel_path.stem + actionplan_excel_sheet_name + "_segmented.pkl")

    if segmented_pkl.exists() and segmented_pkl.stat().st_size > 0 and not refresh_pickle:
        print(f"exist {segmented_pkl}")
        return str(segmented_pkl)

    print(f"new file loading: {excel_path}")

    sheet = pd.read_excel(excel_path, sheet_name=actionplan_excel_sheet_name)

    sentences = []
    if target_col not in sheet.columns:
        raise ValueError(f"[WARN] '{target_col}' not in sheet: {actionplan_excel_sheet_name}")

    col_values = sheet[target_col].dropna().astype(str).str.strip()
    sentences.extend([s for s in col_values if s])

    with open(segmented_pkl, "wb") as f:
        pickle.dump(sentences, f)

    sleep(0.2)
    print("GET SENTENCE MODE: actionplan(excel, col=文章切り抜き) DONE")
    return str(segmented_pkl)


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
        info_yaml_path = f'{ROOT}/src_rq2/inputmaterial_info.yaml'
        with open(info_yaml_path, 'r', encoding='utf-8') as f:
            lecture_info = yaml.safe_load(f)
        lecture_info_city = lecture_info[str(city_name)]

        #Youtube
        youtube_city_path = f"/Users/rintrin/codes/emorilab_climate_assembly/db_youtube_txt/{city_name}"
        if not os.path.exists(youtube_city_path):
            os.makedirs(youtube_city_path, exist_ok=True)
        new_lecture_info_city = lecture_info_city
        for lecture_key, lecture_each_info in lecture_info_city.items():
            YOUTUBE_URL = lecture_each_info.get("Youtube_path")
            yt_txt_path = lecture_each_info.get("Youtube_txt_path")
            if yt_txt_path is None:
                yt_txt_path = f"{lecture_key}_youtube_txt.txt"
                lecture_each_info["Youtube_txt_path"] = yt_txt_path
            yt_txt_path = os.path.join(youtube_city_path, yt_txt_path)
            if not os.path.exists(yt_txt_path):
                with open(yt_txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write("")  # 空のファイルとして作成
                print(f"Created: {yt_txt_path}")
            else:
                print(f"Skipped (already exists or missing path): {yt_txt_path}")

            # 既存ASRの確認
            txt = ""
            if os.path.exists(yt_txt_path):
                with open(yt_txt_path, 'r', encoding='utf-8') as f:
                    txt = f.read()
            if txt.strip():
                print(f"{lecture_key} : IS ALREADY DONE")
            elif YOUTUBE_URL:
                presentation_timespan = lecture_each_info.get("Presentation_timespan")
                presentation_timespan = str(presentation_timespan)
                presentation_length_second = transcribe_youtube_to_text(YOUTUBE_URL, presentation_timespan=presentation_timespan, output_path=yt_txt_path, language='ja')
                lecture_each_info["PresentationLengthSecond"] = presentation_length_second
            
            # 句読点挿入（idempotent）
            PUNCTUATED_OUTPUT_PATH = yt_txt_path.replace(".txt", "_punc_added.txt")
            if os.path.exists(PUNCTUATED_OUTPUT_PATH) and os.path.getsize(PUNCTUATED_OUTPUT_PATH) > 0:
                print(f"{PUNCTUATED_OUTPUT_PATH} : IS ALREADY DONE")
            else:
                predict_and_insert_punctuation(yt_txt_path)
            
            new_lecture_info_city[lecture_key] = lecture_each_info
        
        # city_name 部分だけ更新（無ければ追加）
        lecture_info[city_name] = lecture_info_city
        # 書き戻し
        with info_yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                lecture_info,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        
    #     # マージ実行（戻り値なしの実装想定）
    #     merge_materialtext_and_youtubetext(city_name, ROOT)

    #     # マージ後TXTを拾う（例：/db_merged_txt/{city}/*.txt）
    #     merged_dir = Path(f"{ROOT}/db_merged_txt/{city_name}")
    #     if merged_dir.exists():
    #         merged_txts = sorted(str(p) for p in merged_dir.glob("*.txt"))
    #         if merged_txts:
    #             target_txt_files = merged_txts  # 以降はマージ後TXTのみを使用
        
        

    # # 3) 文分割 -> PKL（TXTと同ディレクトリに *_segmented.pkl を作成）
    segmented_pkl_pth_list = []
    # bunkai = Bunkai()

    # for txt_pth in target_txt_files:
    #     if not os.path.exists(txt_pth):
    #         print(f"[WARN] TXT not found: {txt_pth}")
    #         continue

    #     with open(txt_pth, 'r', encoding='utf-8') as f:
    #         text_data = f.read()

    #     # 文分割
    #     segmenter_list = list(bunkai(text_data))
    #     # 改行除去など軽い正規化
    #     segmenter_list = [s.replace('\n', '') for s in segmenter_list]

    #     # PKL 保存（TXTと同じ場所に *_segmented.pkl）
    #     segmented_pkl = txt_pth.replace(".txt", "_segmented.pkl")
    #     with open(segmented_pkl, 'wb') as f:
    #         pickle.dump(segmenter_list, f)

    #     segmented_pkl_pth_list.append(segmented_pkl)
    #     sleep(0.2)

    # ★ ここで PKL のリストを返す（calc_similarity_ja が想定）
    print(f"GET SENTENCE MODE:{mode} DONE")
    return segmented_pkl_pth_list


def get_sentences_annotated(city_name, mode="inputmaterial", refresh_pickle=False):
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
    split_pat = r"[。\.]+"

    for txt_pth in target_txt_files:
        if not os.path.exists(txt_pth):
            print("PATH NOT EXIST")
            continue

        segmented_pkl = txt_pth.replace(".txt", "_segmented.pkl")
        if os.path.exists(segmented_pkl) and os.path.getsize(segmented_pkl) > 0 and not refresh_pickle:
            segmented_pkl_pth_list.append(segmented_pkl)
            print(f"exist {segmented_pkl}")
            continue

        print(f"new file loading: {txt_pth}")
        with open(txt_pth, "r", encoding="utf-8", errors="ignore") as f:
            text_data = f.read()

        sentences = re.findall(r"[^。]+。|[^。]+$", text_data)
        sentences = [s.replace("\n", "").strip() for s in sentences if s.strip()]

        with open(segmented_pkl, "wb") as f:
            pickle.dump(sentences, f)

        segmented_pkl_pth_list.append(segmented_pkl)
        sleep(0.2)

    print(f"GET SENTENCE MODE:{mode} DONE")
    return segmented_pkl_pth_list
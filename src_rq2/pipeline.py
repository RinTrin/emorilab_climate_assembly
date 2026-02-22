import os
import json
import yaml
import subprocess

from calc_similarity import calc_similarity_ja

from input_organize import get_sentences, get_sentences_annotated, get_sentences_actionplan_excel
from analyses import actor_analysis, presentation_length_analysis #, expert_words_analysis
from utils import filter_top1_score, add_presenter_columns_to_analyzed_csv, pickle_load

from political_analysis import political_analysis

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"

def pipeline(city_name=None):
    """
    まず、pdfから文字起こししたテキストとyoutubeから文字起こしした文章をそのまま統合。
    その後、初等の文分解関数Bunkai()を用いて、文のリストをpickleに格納。
    これらを情報提供とアクションプランの両方で実施。
    DONE: 櫻井さんに、youtubeからの文字起こしを直してもらう。4B1とアクションプランだけ。
    TODO: 一人の講演者に対して2つのインプット資料がある時の対処を決める。
    """
    input_sentences_pkl = get_sentences_annotated(city_name, mode="inputmaterial", refresh_pickle=True)  # ← マージ後TXTを使って分割
    # action_sentences_pkl = get_sentences(city_name, mode="actionplan")    # ← PDF由来TXTのみで分割
    actionplan_excel_sheetname="comprehensive" # comprehensiveかdeduplication
    # actionplan_excel_sheetname = "deduplication"
    action_sentences_pkl = get_sentences_actionplan_excel("/Users/rintrin/codes/emorilab_climate_assembly/db_txt/Atugi/actionplan/厚木アクションプラン_本案_要求文整理v2.xlsx", refresh_pickle=True, actionplan_excel_sheet_name=actionplan_excel_sheetname) 
    # action_sentences_pkl = ["/Users/rintrin/codes/emorilab_climate_assembly/db_txt/Atugi/actionplan/厚木アクションプラン要求文整理_富田一部修正_segmented.pkl"]
    action_sentences_pkl = [action_sentences_pkl]
    print(action_sentences_pkl)
    print("Action Done")
    # ghjkl
    
    """
    続いて、SentenceBertJapanese(芝山&新納, 2021; https://ipsj.ixsq.nii.ac.jp/record/212207/files/IPSJ-NL21249007.pdf)を使用。
    論文内では、使用モデルは「東北大版 BERT と NICT 版 BERT から構築した SentenceBERT が同程度の高い性能を示した」とある中で、利用可能性と評判？から、cl-tohoku/bert-base-japaneseモデルを使用。
    1文ずつベクトル化したのち、アクションプランの1文に対してコサイン類似度が高い情報提供の文3つを特定。
    TODO: 何時限ベクトルに変換しているかを確認する。
    """
    # calculate similarity（PKLのリストを渡す）
    similarity_add_csv = calc_similarity_ja(
        action_plans_pkl_pth_list=action_sentences_pkl,
        input_materials_pkl_pth_list=input_sentences_pkl
    )
    # similarity_add_csv = "/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files/analyzed_top3_across_all_files_ja.csv"
    
    presenter_role_dict = return_presenter_role_dict(city_name)
    
    """
    上位3つも用意したものの、使い道がなかったため、上位1文に絞り込み。
    """
    analyzed_csv_pth = filter_top1_score(similarity_add_csv)
    add_presenter_columns_to_analyzed_csv(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname)
    # info for analyses
    # print("PREPRERE", presenter_role_dict)
    
    """
    注目はあまりされていませんが、日本語の政治文書を、右派左派で数値化した論文が2024年に出版されている。（https://arxiv.org/pdf/2405.07320 ）
    従来手法は議会や政党全体の政治的イデオロギーを分析するにとどまっているため、個々人の政治的イデオロギーを分析するにはこちらの手法が適していた。
    ただし、sentence-bertの事前学習については実施しない。確かに意見文だけを抽出することによって、ベクトル表現の特徴がより強く反映されるように思うし、意見文とそれ以外の事実文などの比率は人によりけりなため、意見文だけを抽出した方がいいだろう。
    しかし、そういったことをしない論文もあるようだし、ここではその意見文かどうかが重要には思えない。よって、ここでは採用しない。
    TODO: 上記の実装をする。現状上記の実装をしておらず、簡易的な実験としてbertを用いたに限る。
    TODO: 実装ができたら、ベンチマークの作り方を考える。
    本研究での限界に、議員の発言を平均ベクトルで表現していることから、時間変化を反映できない点が挙げられていたが、今回の分析対象である気候市民会議は各情報提供者の発言時間は1時間前後であり、日を跨いだ時間変化が生じないため、問題ないと考えられる。
    しかし、LLM自体の学習データの偏りは以前不明であり、ここは未解決のままである。
    これらの点に留意しつつ、人手を解さずに政治的立場を自動推定することができた。"""
    political_analysis(analyzed_csv_pth, presenter_role_dict, input_materials_pkl_pth_list=input_sentences_pkl, actionplan_excel_sheetname=actionplan_excel_sheetname)
    """
    情報提供者のアクターと情報提供した日にちによって、どの程度アクションプランに参照されたかを観察するための分析。
    この考えに至った背景的論文は、、、あったか？ TODO:論文まとめを読む。
    """
    actor_analysis(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname=actionplan_excel_sheetname)
    
    """
    情報提供の発表時間によって、どの程度アクションプランに参照されたかを観察するための分析。
    実験的意味合いが強い。
    """
    presentation_length_analysis(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname=actionplan_excel_sheetname)
    fghjk
    """
    情報提供の専門的用語の多さによって、どの程度アクションプランに参照されたかを観察するための分析。
    実験的意味合いが強い。
    """
    expert_words_analysis(analyzed_csv_pth, presenter_role_dict)


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


if __name__ == '__main__':
    city_name='Atugi'
    print(f'City name: {city_name}')
    if city_name is None:
        print("No City Name")
        exit()
    pipeline(city_name=city_name)

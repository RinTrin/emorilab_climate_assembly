import os
import json
import yaml
import subprocess
import pickle

from calc_similarity import calc_similarity_ja

from input_organize import get_sentences, get_sentences_annotated, get_sentences_actionplan_excel
from analyses import actor_analysis, presentation_length_analysis #, expert_words_analysis
from utils import filter_top1_score, add_presenter_columns_to_analyzed_csv, pickle_load, return_presenter_role_dict

from gpt_similarity import select_similar_sentence

from political_analysis import political_analysis

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"

def pipeline(city_name=None):

    input_sentences_pkl = get_sentences_annotated(city_name, mode="inputmaterial", refresh_pickle=True)  # ← マージ後TXTを使って分割
    actionplan_excel_sheetname="comprehensive" # comprehensiveかdeduplication
    # actionplan_excel_sheetname = "deduplication"
    action_sentences_pkl = get_sentences_actionplan_excel("/Users/rintrin/codes/emorilab_climate_assembly/db_txt/Atugi/actionplan/厚木アクションプラン_本案_要求文整理v2.xlsx", refresh_pickle=True, actionplan_excel_sheet_name=actionplan_excel_sheetname) 
    action_sentences_pkl = [action_sentences_pkl]
    
    presenter_role_dict = return_presenter_role_dict(city_name)
    
    analyzed_csv_pth = select_similar_sentence(
        action_sentences_pkl,
        input_sentences_pkl,    # 複数
        presenter_role_dict,
        city_name,
        actionplan_excel_sheetname,
        model="gpt-5.2-2025-12-11",
        temperature=0.0,
        out_csv_path=f"{ROOT}/analysis_results/each_sentence_all_files/gpt_check_{actionplan_excel_sheetname}.csv",    
        )
    print(analyzed_csv_pth)
    
    political_analysis(analyzed_csv_pth, presenter_role_dict, input_materials_pkl_pth_list=input_sentences_pkl, actionplan_excel_sheetname=actionplan_excel_sheetname)
    actor_analysis(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname=actionplan_excel_sheetname)
    presentation_length_analysis(analyzed_csv_pth, presenter_role_dict, actionplan_excel_sheetname=actionplan_excel_sheetname)
    fghjk
    """
    情報提供の専門的用語の多さによって、どの程度アクションプランに参照されたかを観察するための分析。
    実験的意味合いが強い。
    """
    # expert_words_analysis(analyzed_csv_pth, presenter_role_dict)

if __name__ == '__main__':
    city_name='Atugi'
    print(f'City name: {city_name}')
    if city_name is None:
        print("No City Name")
        exit()
    pipeline(city_name=city_name)

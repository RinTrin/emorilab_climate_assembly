# import functools

# from ja_sentence_segmenter.common.pipeline import make_pipeline
# from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
# from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
# from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

from bunkai import Bunkai
from pathlib import Path
import pickle
from time import sleep

def preprocess(output_txt_file_list):
    
    # segmenterの準備
    # split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
    # concat_tail_te = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(て)$", remove_former_matched=False)
    # segmenter = make_pipeline(normalize, split_newline, concat_tail_te, split_punc2)
    

    segmented_pkl_pth_list = []
    
    for output_txt_file in output_txt_file_list:
        
        segmented_txt_file = f'{output_txt_file.replace("db_txt", "db_pdf").replace(".txt", "")}_segmented.pkl'
        
        # method(output_txt_file, segmented_txt_file)
        
        segmented_pkl_pth_list.append(segmented_txt_file)
        sleep(1)  # 1秒待つ
    
    return segmented_pkl_pth_list

def method(output_txt_file, segmented_txt_file):
    
    bunkai = Bunkai(path_model=Path("bunkai-model-directory"))
    
    with open(output_txt_file, 'r') as f:
        text_data = f.read()
    
    # 文ごとに分割
    segmenter_list = list(bunkai(text_data))
    # segmenter_list = list(segmenter(text_data))
    # print(segmenter_list)
    
    
    # いらない単語を削除
    new_segmenter_list = []
    for segmenter in segmenter_list:
        segmenter = segmenter.replace('\n', '')
        new_segmenter_list.append(segmenter)
        
    with open(segmented_txt_file, 'wb') as f:
        pickle.dump(new_segmenter_list, f)
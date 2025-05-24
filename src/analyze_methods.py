import pickle
import os
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict
import numpy as np
import torch.nn.utils.rnn as rnn

def analyze(action_plans_pkl_pth_list, input_materials_pkl_pth_list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    save_dict = defaultdict(dict)
    save_cosine_scores = []
    
    for action_pkl_pth in action_plans_pkl_pth_list:
        action_data = pickle_load(action_pkl_pth)
        print(action_data)
    
    for action_pkl_pth in action_plans_pkl_pth_list:
        action_data = pickle_load(action_pkl_pth)
        action_embeddings = model.encode(action_data, convert_to_tensor=True)
        for input_pkl_pth in input_materials_pkl_pth_list:
            input_data = pickle_load(input_pkl_pth)
            input_embeddings = model.encode(input_data, convert_to_tensor=True)
            
            cosine_scores = util.cos_sim(action_embeddings, input_embeddings)
            save_cosine_scores.append(cosine_scores)
            
            m, n = cosine_scores.size()
            argmax_list = torch.topk(cosine_scores.flatten(), 10).indices.tolist()
            for argmax_index in argmax_list:
                print(argmax_index)
                m_a = argmax_index // n
                n_a = argmax_index % n
                print(cosine_scores[m_a][n_a])
                print(action_data[m_a])
                print(input_data[n_a])
                print()
            
            ### テキスト全体で類似度を計算する
            cosine_scores_ave = torch.mean(cosine_scores)
            save_dict[os.path.basename(action_pkl_pth).replace("_segmented.pkl", "")][os.path.basename(input_pkl_pth).replace("_segmented.pkl", "")] = cosine_scores_ave.item()
    
    # 類似度の高い組み合わせを取得
    pad_tensor = rnn.pad_sequence(save_cosine_scores, batch_first=True)
    print(pad_tensor.size())
    argmax_list = torch.argmax(pad_tensor).tolist()
    
    csv_pth = os.path.join(Path(__file__).parents[1], "output/analyzed_each_text_file.csv")
    df = pd.DataFrame(save_dict).sort_index()
    df.to_csv(csv_pth)
    
    return csv_pth


def pickle_load(pkl_pth):
    with open(pkl_pth, 'rb') as f:
        return pickle.load(f)
import os
import pandas as pd
from data_methods import get_data
from preprocess_methods import preprocess
# from analyze_methods import analyze
from analyze_methods_each import analyze_simmality, analyze_political_leaning
from summarize_methods import summarize
from visualize_methods import visualize, plot_political_bubble

def pipeline(city_name=None):
    if city_name is None:
        exit()
    
    print(f'City name: {city_name}')
    presenter_role_dict = return_presenter_role_dict()
    
    # Get the data of the city
    action_plans = get_data(city_name, mode='actionplan')
    input_materials = get_data(city_name, mode='inputmaterial')
    
    # Preprocess the data
    action_plans_pkl_pth_list = preprocess(action_plans)
    input_materials_pkl_pth_list = preprocess(input_materials)
    
    # Analyze the similarity of data
    # analyzed_csv_pth = analyze_simmality(action_plans_pkl_pth_list, input_materials_pkl_pth_list)
    # analyzed_csv_pth = analyze(action_plans_pkl_pth_list, input_materials_pkl_pth_list)
    analyzed_csv_pth = "/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files/analyzed_top3_across_all_files.csv"
    
    # Analyze the political vias of data
    policital_dir = os.path.join("/Users/rintrin/codes/emorilab_climate_assembly/analysis_results", "political_analysis")
    os.makedirs(policital_dir, exist_ok=True)
    # df_political, df_political_csv_path = analyze_political_leaning(analyzed_csv_pth, presenter_role_dict, save_dir=policital_dir)
    df_political = pd.read_csv("/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/political_analysis/presenter_political_leaning.csv")
    
    # Summarize by source and role
    # summarize_top1_by_source_pth, summarize_top1_by_role_pth = summarize(analyzed_csv_pth, presenter_role_dict)
    
    # Visualize the data
    # visualize(analyzed_csv_pth)
    plot_political_bubble(df_political, save_path=os.path.join(policital_dir, "political_bubble_plot.png"))
    
def return_presenter_role_dict():
    presenter_role_dict = {
    # public private adademic citizen
    'lecture1_1_segmented.pkl': {'Presenter': '江守正多', 'Role': 'academic'}, #NIES
    'lecture1_2_segmented.pkl': {'Presenter': '渡部厚志', 'Role': 'academic'}, #IGES
    #  'lecture2_1_segmented.pkl': {'Presenter': '厚木市', 'Role': 'public'}, #前回の振り返りや渡部厚志氏からの補足説明 #Driveには含まれていない。
    'lecture2_1_segmented.pkl': {'Presenter': '新井聡史', 'Role': 'public'}, #神奈川県気候適応センター
    'lecture2_2_segmented.pkl': {'Presenter': '厚木市', 'Role': 'public'}, #厚木市まちづくり計画部長 前場徹
    'lecture2_3_segmented.pkl': {'Presenter': '厚木市', 'Role': 'public'}, #厚木市環境政策課 山﨑尚裕
    'lecture2_4_segmented.pkl': {'Presenter': '厚木市', 'Role': 'public'}, #上記の追加資料
    #  'lecture3_1_segmented.pkl': {'Presenter': '厚木市', 'Role': 'public'}, #振り返り。 #Driveには含まれていない。
    'lecture3_1_segmented.pkl': {'Presenter': '松原弘直', 'Role': 'academic'}, #環境エネルギー政策研究所
    'lecture3_2_segmented.pkl': {'Presenter': '梶田佳孝', 'Role': 'academic'}, #東海大学建築都市学部
    'lecture3_3_segmented.pkl': {'Presenter': '山本佳嗣', 'Role': 'academic'}, #東京工芸大学 工学部工学科
    'lecture3_4_segmented.pkl': {'Presenter': '村上千里', 'Role': 'citizen'}, #公益社団法人 日本消費生活アドバイザー・コンサルタント・相談員協会
    'lecture4A_1_segmented.pkl': {'Presenter': '遠藤睦子', 'Role': 'citizen'}, #一般社団法人 あつぎ市民発電所
    'lecture4A_2_segmented.pkl': {'Presenter': '木原浩貴', 'Role': 'private'}, #木原浩貴 #TODO:要検討
    'lecture4A_3_segmented.pkl': {'Presenter': '小田原市', 'Role': 'public'}, #小田原市環境部 ゼロカーボン推進課
    'lecture4A_4_segmented.pkl': {'Presenter': '日産自動車株式会社', 'Role': 'private'}, #日産自動車株式会社
    'lecture4A_6_segmented.pkl': {'Presenter': '神奈川中央交通株式会社', 'Role': 'private'}, #神奈川中央交通株式会社
    'lecture4B_1_segmented.pkl': {'Presenter': '一社 あつぎ市民発電所', 'Role': 'citizen'}, #一社)あつぎ市民発電所 #TODO:要検討
    'lecture4B_2_segmented.pkl': {'Presenter': 'コムアソシエイツ', 'Role': 'citizen'}, #コムアソシエイツ
    'lecture4B_3_segmented.pkl': {'Presenter': '青砥航次', 'Role': 'citizen'}, #NPO法人神奈川自然保護協会
    'lecture4B_4_segmented.pkl': {'Presenter': '山本佳嗣', 'Role': 'academic'}, #東京工芸大学 工学部工学科
    'lecture4B_5_segmented.pkl': {'Presenter': '浅利美鈴', 'Role': 'academic'}, #ちきゅう研
    'lecture4B_6_1_segmented.pkl': {'Presenter': '八一農園', 'Role': 'citizen'}, #TODO:要検討
    'lecture4B_6_2_segmented.pkl': {'Presenter': '八一農園', 'Role': 'citizen'}, 
    'lecture4B_7_segmented.pkl': {'Presenter': '厚木市', 'Role': 'public'}, #厚木市環境事業課
    }
    #TODO: lecture2_4を、2_3_2に書き換える。
    
    return presenter_role_dict
    

if __name__ == '__main__':
    # pipeline(city_name='多摩市')
    pipeline(city_name='厚木')
    


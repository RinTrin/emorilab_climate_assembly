
import os

from data_methods import get_data
from preprocess_methods import preprocess
from analyze_methods import analyze
from visualize_methods import visualize



def pipeline(city_name=None):
    if city_name is None:
        exit()
    
    print(f'City name: {city_name}')
    
    # Get the data of the city
    action_plans = get_data(city_name, mode='actionplan')
    input_materials = get_data(city_name, mode='inputmaterial')
    

if __name__ == '__main__':
    import argparse

    # ArgumentParserのインスタンスを作成
    parser = argparse.ArgumentParser(description="コマンドライン引数の例")

    # 引数を追加
    parser.add_argument("input_value", help="入力値")  # 必須引数
    parser.add_argument("--optional", help="オプション引数", default="デフォルト値")  # オプション引数

    # 引数を解析
    args = parser.parse_args()

    # 引数を表示
    print("使用する市:", args.input_value)
    print("オプションの値:", args.optional)
    
    city_names = []
    if "/" in args.input_value:
        city_names = args.input_value.split("/")
    else:
        city_names.append(args.input_value)
    # print(city_names)
        
    for city_name_use in city_names:
        pipeline(city_name=str(city_name_use))
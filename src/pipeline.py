

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
    
    # Preprocess the data
    action_plans_pkl_pth_list = preprocess(action_plans)
    input_materials_pkl_pth_list = preprocess(input_materials)
    
    # Analyze the data
    analyzed_csv_pth = analyze(action_plans_pkl_pth_list, input_materials_pkl_pth_list)
    
    # Visualize the data
    visualize(analyzed_csv_pth)
    
    
    

if __name__ == '__main__':
    # pipeline(city_name='多摩市')
    pipeline(city_name='厚木市')
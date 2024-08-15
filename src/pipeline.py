

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
    output_txt_file_list = get_data(city_name)
    
    # Preprocess the data
    pkl_pth_list = preprocess(output_txt_file_list)
    
    # Analyze the data
    analyze(pkl_pth_list)
    
    # Visualize the data
    visualize(data)
    
    
    

if __name__ == '__main__':
    pipeline(city_name='多摩市')
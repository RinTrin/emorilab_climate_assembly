
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
    # pipeline(city_name='多摩市')
    pipeline(city_name='厚木市')
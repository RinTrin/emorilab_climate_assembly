
import os
import subprocess
from pathlib import Path

def get_data(city_name, mode='actionplan'):
    
    output_txt_file_list = []
    
    city_folder_pth = os.path.join(Path(__file__).parents[1], f'db/{city_name}')
    if mode == 'actionplan':
        city_folder_pth = os.path.join(city_folder_pth, 'actionplan')
    elif mode == 'inputmaterial':
        city_folder_pth = os.path.join(city_folder_pth, 'inputmaterial')
    
    for city_file_pth in os.listdir(city_folder_pth):
        if city_file_pth.endswith('.pdf'):
            city_file_pth = os.path.join(city_folder_pth, city_file_pth)
            output_file_pth = f'{city_file_pth.replace(".pdf", "")}.txt'
            if os.path.exists(output_file_pth):
                output_txt_file_list.append(output_file_pth)
                continue
            cmd = f"bash pdf2text_mac_default.sh {city_file_pth} >> {output_file_pth}"
            print(cmd)
            subprocess.call(cmd, shell=True)
            output_txt_file_list.append(output_file_pth)
    
    return output_txt_file_list
            
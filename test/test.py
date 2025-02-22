

import os
import subprocess

dir_path = "/Users/rintrin/codes/emorilab_climate_assembly/"

for path in os.listdir(os.path.join(dir_path, "db_pdf")):
    
    if path =="not_used_みその" or path==".DS_Store":
        continue
    
    for action_or_input in ["actionplan", "inputmaterial"]:
        for file in os.listdir(os.path.join(dir_path, "db_pdf", path, action_or_input)):
            if file.endswith(".txt"):
                cmd = f"mv {os.path.join(dir_path, 'db_pdf', path, action_or_input, file)} {os.path.join(dir_path, 'db_txt', path, action_or_input, file)}"
                subprocess.call(cmd, shell=True)
                # print(cmd)
                # hjkl
        
    
    # dir_make_path = os.path.join(dir_path, "db_txt", path)
    # os.makedirs(dir_make_path)
    
    
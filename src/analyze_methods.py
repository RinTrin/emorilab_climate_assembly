import pickle


def analyze(pkl_pth_list):
    
    for pkl_pth in pkl_pth_list:
        with open(pkl_pth, 'r') as f:
            segmenter_list = pickle.load(f)
        
        for segmenter in segmenter_list:
            print(segmenter)
    

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize(csv_pth):
    
    df = pd.read_csv(csv_pth, index_col=0)
    print(df)
    
    
    plt.rcParams['figure.subplot.left'] = 0.3
    svm = sns.heatmap(df, annot=True, cmap='coolwarm')
    
    figure = svm.get_figure()    
    timestr = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    figure.savefig(os.path.join(Path(__file__).parents[1], f'output/imgs/svm_conf_{timestr}.png'), dpi=400)
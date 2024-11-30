
import pandas as pd
import matplotlib as plt
import nlplot

#データファイルを読み込む
df1 = pd.read_csv('/Users/rintrin/codes/emorilab_climate_assembly/db/test/厚木気候市民会議すべて.csv')

import MeCab

#文章を分解し、名詞のみを取り込む
def mecab_text(text):

    #単語を分割する
    tagger = MeCab.Tagger("")
    tagger.parse('')
    node = tagger.parseToNode(text)

    #名詞を格納するリスト
    word_list = []

    while node:
        #node.feature.split(',')[0]で品詞が抽出できる
        word_type = node.feature.split(',')[0]
        if word_type == '名詞':
            #名詞のみword_listに入れる
            if (node.surface != "こと") and (node.surface != "ところ"):
                word_list.append(node.surface)
        #次のノードへ行く
        node = node.next
    return word_list

#形態素結果をリスト化し、データフレームdf1に結果を列追加する
df1['words'] = df1['transcription'].apply(mecab_text)


df1 = df1[['transcription','words']]

# target_colを'words'で指定する
npt = nlplot.NLPlot(df1, target_col='words')

# top_nで頻出上位単語, min_freqで頻出下位単語を指定できる
# ストップワーズを設定
stopwords = npt.get_stopword(top_n=0, min_freq=0)


from plotly.offline import iplot

# ビルド（データ件数によっては処理に時間を要します）
npt.build_graph(stopwords=stopwords, min_edge_frequency=20)

# ビルド後にノードとエッジの数が表示される。ノードの数が100前後になるようにするとネットワークが綺麗に描画できる
#>> node_size:63, edge_size:63

fig_co_network = npt.co_network(
    title='Co-occurrence network',
    sizing=100,
    node_size='adjacency_frequency',
    color_palette='hls',
    width=1100,
    height=700,
    save=False
    )
iplot(fig_co_network)
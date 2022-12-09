# from lib2to3.pgen2.literals import simple_escapes
import pickle
import pandas as pd

#load dataset
dataset = pickle.load(open('model/rec_wisata.pkl', 'rb'))
dataset = pd.DataFrame(dataset)

#load model


def recommendation(nama):
    simi = pickle.load(open('model/similarity.pkl', 'rb'))
    indices = pd.Series(dataset.index)
    # indices[:10]
    recommended_wisata = []
    # Mengambil nama wisata berdasarkan variabel indicies
    idx = indices[indices == nama].index[0]
    # Membuat series berdasarkan skor kesamaan
    score_series = pd.Series(simi[idx]).sort_values(ascending = False)
    # mengambil index dan dibuat 10 baris rekomendasi terbaik
    top_10_indexes = list(score_series.iloc[1:11].index)
    for i in top_10_indexes:
        recommended_wisata.append(list(dataset.index)[i])
        
    return recommended_wisata

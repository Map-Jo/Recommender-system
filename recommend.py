import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kmr_dataset import load_rates
from kmr_dataset import get_paths
import koreanize_matplotlib
from kmr_dataset import load_histories
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title('Hello!')
st.header('May I recommend you a Movie?')
paths = get_paths(size='small')

rates, timestamps = load_rates(size='small')


histories = load_histories(size='small')

path = get_paths(size='small')[3]

movie_df = pd.read_table(path)
movie = movie_df[['movie', 'title', 'title_eng']]

genres_df = pd.read_csv(paths[2])
genres_df = genres_df.groupby("movie").agg({"genre" : lambda x : '/'.join(x)})
movie_genre_df = movie.merge(genres_df, on='movie', how='left')
movie_genre_df = movie_genre_df.dropna()

rates_df = pd.read_csv(paths[5])

df = movie_df[movie_df['title_eng'].notnull()].copy()

tfidfvect = TfidfVectorizer()
tfidf_title = tfidfvect.fit_transform(df['title_eng'])
tfidfvect.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_title.toarray(), columns=tfidfvect.get_feature_names_out())

cosine_matrix = cosine_similarity(tfidf_title, tfidf_title)

def find_movie(title, sim_matrix, df):
    try:
        movie_id = df.loc[df["title_eng"].notnull() & 
                          df["title_eng"].str.contains(title), "title"].index[0]
        df_sim = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)
        sim = df_sim[movie_id].nlargest(10)
        df_sim = df.loc[sim.index, ['movie','title', 'title_eng']].join(sim)
        return df_sim
    except:
        return "추천할 영화 없음"
title = st.text_input('Movie title')
st.write('The Recommend movie title is', find_movie(title, cosine_matrix, df))
# find_movie(title, cosine_matrix, df)

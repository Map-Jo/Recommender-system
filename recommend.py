import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import koreanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.title('Hello!')
st.header('Are you looking for this movie?')

image = Image.open('movie_night.jpg')
st.image(image)

movie_df = pd.read_table('movies.txt')
movie = movie_df[['movie', 'title', 'title_eng']]

genres_df = pd.read_csv('genres.csv')
genres_df = genres_df.groupby("movie").agg({"genre" : lambda x : '/'.join(x)})
movie_genre_df = movie.merge(genres_df, on='movie', how='left')
movie_genre_df = movie_genre_df.dropna()

rates_df = pd.read_csv('rates.csv')

df = movie_df[movie_df['title'].notnull()].copy()

tfidfvect = TfidfVectorizer()
tfidf_title = tfidfvect.fit_transform(df['title'])
tfidfvect.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_title.toarray(), columns=tfidfvect.get_feature_names_out())

cosine_matrix = cosine_similarity(tfidf_title, tfidf_title)

def find_movie(title, sim_matrix, df):
    try:
        movie_id = df.loc[df["title"].notnull() & 
                          df["title"].str.contains(title), "title"].index[0]
        df_sim = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)
        sim = df_sim[movie_id].nlargest(5)
        df_sim = df.loc[sim.index, ['movie','title', 'title_eng']].join(sim)
        return df_sim
    except:
        return "추천할 영화 없음"
title = st.text_input('Movie title')
st.write('The Recommend movie title is', find_movie(title, cosine_matrix, df))

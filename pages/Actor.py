import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import koreanize_matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.title('Hello!')
st.header('Who\'s in the cast?')
image = Image.open('actor.jpg')
st.image(image)

movies_df = pd.read_table('data/movies.txt')
catings_df = pd.read_csv('data/castings.csv')
peoples_df = pd.read_table('data/peoples.txt')
rates_df = pd.read_csv('data/rates.csv')

rate = rates_df.groupby('movie')['rate'].agg('mean')

act = catings_df.merge(peoples_df, on='people', how='left')
act = act[['movie', 'korean', 'leading']]
act = movies_df.merge(act, on='movie', how='left')
act = act.merge(rate, on='movie', how='left')
act = act[['title', 'korean', 'rate','grade']]
title = st.text_input('Actor or Actress Name')
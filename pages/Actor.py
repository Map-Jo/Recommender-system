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

movies_df = pd.read_parquet('data/moives.parquet.gzip')
castings_df = pd.read_parquet('data/castings.parquet.gzip')
peoples_df = pd.read_parquet('data/peoples.parquet.gzip')
rates_df = pd.read_parquet('data/rates.parquet.gzip')

rate_mean = rates_df.groupby('movie')['rate'].mean()
rate_mean = pd.DataFrame(rate_mean).reset_index()
rate_mean = rate_mean.rename(columns={'movie':'movie',
                                    'rate':'평점'})

act = castings_df.merge(peoples_df, on='people', how='left')
act = act[['movie', 'korean', 'leading']]
act = movies_df.merge(act, on='movie', how='left')
act = act.merge(rate_mean, on='movie', how='left')

act = act[['title', 'korean', '평점','grade']]
act = act[(act['korean'].notnull()) & act['평점'].notnull()]
act = act.sample(10000, random_state=42).copy()

tfidfvect = TfidfVectorizer()
tfidf_name = tfidfvect.fit_transform(act['korean'])
tfidfvect.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_name.toarray(), columns=tfidfvect.get_feature_names_out())

cosine_matrix = cosine_similarity(tfidf_name, tfidf_name)

def find_movie(name, sim_matrix, df):
    try:
        actor_name = act.loc[act["korean"].notnull() & 
                          act["korean"].str.contains(name), "korean"].index[0]
        df_sim = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)
        sim = df_sim[actor_name].nlargest(5)

        df_sim = act.loc[sim.index, ['title', 'korean', '평점','grade']].join(sim)
        return df_sim
    except:
        return "찾으시는 영화배우와 관련된 영화 검색 결과가 없습니다."

name = st.text_input('Actor or Actress name')
st.write('The Recommend movie title is', find_movie(name, cosine_matrix, act))

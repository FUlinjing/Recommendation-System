import os
import pandas as pd
import numpy as np
#from recommender_app.config import RESOURCES_FOLDER
#from recommender_app.utils.recommendations_svd import RecommenderSvd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import altair as alt
from PIL import Image

#global recommender
#global recommender_list

st.write ( "# Simple Recommendation App")
image=Image.open('pic.png')
st.image(image, use_column_width=True)

st.header('Enter Your Favorate Movie')

movie= "Shaolin Soccer (Siu lam juk kau) (2001)"

sequence =st.text_area("write your movie here",movie, height=2)

st.header ('output')

latent_matrix_1_mvp=pd.read_csv('latent_matrix_small.csv', index_col=0)


def movieExists(movie_name,df):
    if movie_name in df.index:
        return True
    else:
        return False

def find_similar_movie(movie_name):
    if movieExists(movie_name,latent_matrix_1_mvp):
        a_1 = np.array(latent_matrix_1_mvp.loc[movie_name]).reshape(1, -1)
        score_1 = cosine_similarity(latent_matrix_1_mvp, a_1).reshape(-1)
        dictDf = {'content': score_1} 
        similar = pd.DataFrame(dictDf, index = latent_matrix_1_mvp.index)
        similar.sort_values('content', ascending = False, inplace = True)
        return similar[1:].head(5)
    else:
        raise ValueError('ERROR: The Movie is not Recognised')

x=find_similar_movie(sequence)
st.write(x)

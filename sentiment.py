import numpy as np
import pandas as pd
import streamlit as st 
from streamlit_option_menu import option_menu
import plotly.express as px

st.set_page_config(page_title ="Restaurant Review Sentiment Analysis",layout="wide")

st.write("""

<div style='text-align:center'>
    <h1 style='color:#009999;'>Restaurant Review Sentiment Analysis</h1>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
     opt = option_menu("Review Analysis ",
                      ["Info","Analysis"],
                      menu_icon="cast",
                      styles={
                          "container":{"padding":"4!important","background-color":"gray"},
                          "icon":{"color":"red","font-size":"20px"},
                          "nav-link":{"font-size":"20px","text-align":"left"},
                          #"nav-link-selected":{"background-color":"yellow"}
                      })


dataset = pd.read_csv('C:/Users/muges/Downloads/a2_RestaurantReviews_FreshDump.tsv', delimiter = '\t', quoting = 3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


if opt == "Info":
    st.markdown("This project is based on viewing  the review of a restaurant and analysing whether it is positive sentiment or negative sentiment")
    st.markdown("Technologies used: Natural Language Processing Toolkit,Python Scripting,Streamlit")
    
if opt == "Analysis":
    selected_review = st.selectbox("Select any review",options=dataset['Review'])
    corpus=[]

    
    review = re.sub('[^a-zA-Z]', ' ',selected_review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    


    # Loading BoW dictionary
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle
    cvFile='C:/Users/muges/Downloads/c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))

    X_fresh = cv.transform(corpus).toarray()
    


    import joblib
    classifier = joblib.load('C:/Users/muges/Downloads/c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_fresh)
    st.write(y_pred)

    if y_pred == 1:
        st.write('## :green[The review is positive ]')
    elif y_pred == 0:
        st.write('## :red[The review is negative ]')
      
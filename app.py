import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


@st.cache
def load_data_models():

    df = pd.read_csv("data/augmented_datafile.csv")
    lb = LabelEncoder()
    lb.fit(df['Categories'])
    df['Target'] = lb.transform(df['Categories'])
    df['Target'] = lb.transform(df['Categories'])
    model = tf.keras.models.load_model("final_model.h5",custom_objects={'KerasLayer': hub.KerasLayer})
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    kw_model_bert = KeyBERT(model=sbert_model) 
    summarizer = pipeline("summarization") 
    return df,lb,kw_model_bert,summarizer

df,lb,kw_model_bert,summarizer,model = load_data_models()

def text_cleaning(text):
    """
    text: text is string type to be cleaned.
    Function aims to clean the text from punctuation or special character, remove the stop words and lemmatizing the words.
   """

    lemmatizer = WordNetLemmatizer()
    text = text.lower()     # or df["Text"].str.lower()
    # eliminate any character that isn't an alphabet like punctuation and numbers ect...
    text = re.sub("[^a-zA-Z]"," ",text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english')) and len(word)>=2] 
    #words = [word for word in words if len(word)>=2] 

    return " ".join(words)


 
def main():

    option = 'Home Page'
    option = st.sidebar.selectbox('NLP Service',('Home Page','Text Topic', 'Keywords Extraction', 'Text Summarization'))


    if option == 'Home Page':
        st.image("images/Text_Classification_image.jpg", width=700)
        st.markdown("### Welcome To The Multi-class Text Classification Web Application ")
        st.write("In the graph below, you can see a brief description of the data by showing the number of elements in each class.")
        st.image("images/data_plot.png",width=800)
        st.markdown("\n What type of NLP service would you like to use? you can choose one of the options in the sidebar")


    elif option=='Text Topic':
        st.subheader("Enter the text you'd like to analyze.")
        input = st.text_area('Enter text') #text is stored in this variable
        key_txt = kw_model_bert.extract_keywords(input, keyphrase_ngram_range=(1, 2), nr_candidates=15, top_n=10,stop_words='english')
        k = str(list(dict(key_txt).keys()))
        prediction = model.predict([tf.convert_to_tensor([input]) ,tf.convert_to_tensor([k]) ])
        
        
        pred = model.predict(text_cleaning(input))

        st.write("Text Topic predicted")


    elif option == 'Keywords Extraction':
        st.subheader("Enter the text you'd like to analyze.")
        input = st.text_area('Enter text') #text is stored in this variable
        key = kw_model_bert.extract_keywords(input, keyphrase_ngram_range=(1, 2), nr_candidates=15, top_n=10,stop_words='english')
        keys = list(dict(key).keys())
        st.write(keys)

        
    
    else:
        st.subheader("Enter the text you'd like to analyze.")
        input = st.text_area('Enter text') #text is stored in this variable
        st.subheader("Summary")
        
        #result = summarizer(input, min_length=20, max_length=70, do_sample=False)
        #summary = result[0]["summary_text"]


        st.write("summary")





main()
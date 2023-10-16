import streamlit as st
import re
import numpy as np
import pickle as pk
import json
import random
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import joblib


# Load Intent Model
loadedIntentClassifier = load_model('intent_model.h5')
loaded_intent_CV = joblib.load('IntentCountVectorizer.sav')

# Load Entity Model
loadedEntityCV = pk.load(open('EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = joblib.load(open('entity_model.sav', 'rb'))

labelencoder_intent = LabelEncoder()

# Load Intent Label Map and Intents JSON
with open('intents.json') as json_data:
    intents = json.load(json_data)
intent_label_map = {cl: labelencoder_intent.transform([cl])[0] for cl in labelencoder_intent.classes_}

st.title("Agriculture Chatbot")

user_query = st.text_input("Hello! How May I Help You Today? ")
if st.button("Ask"):
    # Handle user query and display responses here
    # You can use the existing code for processing queries and generating responses.
    if user_query:
        query = re.sub('[^a-zA-Z]', ' ', user_query).split(' ')
        ps = PorterStemmer()
        tokenized_query = [ps.stem(word.lower()) for word in query]

    processed_text = loaded_intent_CV.transform([' '.join(tokenized_query)]).toarray()
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)

    for key, value in intent_label_map.items():
        if value == result[0]:
            USER_INTENT = key
            break

    for i in intents['intents']:
        if i['tag'] == USER_INTENT:
            response = random.choice(i['responses'])
            st.write("Chatbot:", response)

    entities = getEntities(tokenized_query)
    token_entity_map = dict(zip(entities, tokenized_query))
    # Display entities or other information as needed

import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import json
import random
import re
from nltk.stem.porter import PorterStemmer
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Title for your Streamlit app
st.title("Agriculture Chatbot")

dataset = pd.read_csv('intent.csv', names=["Intent"])
y = dataset["Intent"]

# Load the intent model and CountVectorizer
loadedIntentClassifier = load_model('intent_model.h5')
loaded_intent_CV = joblib.load('IntentCountVectorizer.sav')

# Load entity CountVectorizer and classifier
loadedEntityCV = pk.load(open('EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = joblib.load(open('entity_model.sav', 'rb'))

# Load intents.json
with open('intents.json') as json_data:
    intents = json.load(json_data)


labelencoder_intent = LabelEncoder()
y = to_categorical(labelencoder_intent.fit_transform(y))

intent_label_map = {cl: labelencoder_intent.transform([cl])[0] for cl in labelencoder_intent.classes_}

def getEntities(query):
    query = loadedEntityCV.transform(query).toarray()
    response_tags = loadedEntityClassifier.predict(query)
    entity_list = [list(entity_label_map.keys())[list(entity_label_map.values()).index(tag)] for tag in response_tags if tag in entity_label_map.values()]
    return entity_list

# Initialize variables
USER_INTENT = ""
entity_label_map = {}

# User input field
user_query = st.text_input("Hello! How May I Be Of Assistance? ")

# Button to ask the chatbot
if st.button("Ask"):
    query = re.sub('[^a-zA-Z]', ' ', user_query).split(' ')
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]

    processed_text = loaded_intent_CV.transform([' '.join(tokenized_query)]).toarray()
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)

    USER_INTENT = ""
    for key, value in intent_label_map.items():
        if value == result[0]:
            USER_INTENT = key
            break

    response = ""
    for i in intents['intents']:
        if i['tag'] == USER_INTENT:
            response = random.choice(i['responses'])

    entities = getEntities(tokenized_query)
    token_entity_map = dict(zip(entities, tokenized_query))

    # Display the chatbot's response
    st.write(f"Bot: {response}")



import streamlit as st
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import re
from nltk.stem import PorterStemmer
import random

# Load the intents data from JSON
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Load the pre-trained intent classifier and count vectorizer
loadedIntentClassifier = load_model('intent_model.h5')
loaded_intent_CV = joblib.load('IntentCountVectorizer.sav')

# Load the pre-trained entity classifier and count vectorizer
loadedEntityCV = joblib.load('EntityCountVectorizer.sav')
loadedEntityClassifier = joblib.load('entity_model.sav')

entity_label_map = {}

with open('data-tags.csv') as csv_file:
    dataset = pd.read_csv(csv_file, names=["word", "label"])
    entity_label_map = dict(zip(dataset['label'], dataset['word']))

USER_INTENT = ""

st.title("Agriculture Chatbot")

user_query = st.text_input("User Query")

if user_query:
    query = re.sub('[^a-zA-Z]', ' ', user_query)

    # Tokenize sentence
    query = query.split(' ')

    # Lemmatizing
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]

    # Recreate the sentence from tokens
    processed_text = ' '.join(tokenized_query)

    # Transform the query using the CountVectorizer
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()

    # Make the prediction
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)

    intent_label_map = {cl: labelencoder_intent.transform([cl])[0] for cl in labelencoder_intent.classes_}

    for key, value in intent_label_map.items():
        if value == result[0]:
            USER_INTENT = key
            break

    for i in intents['intents']:
        if i['tag'] == USER_INTENT:
            st.write(random.choice(i['responses']))

    # Extract entities from text
    entities = getEntities(tokenized_query)

    # Mapping between tokens and entity tags
    token_entity_map = dict(zip(entities, tokenized_query))
    st.write(token_entity_map)

import json
import random
from urllib import response
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

global responses, lemmatizer, tokenizer, le, model, input_shape
input_shape = 7

def load_response():
    global responses
    responses = {}
    with open('dataset/chatbot/dataset_chatbot.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']

def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    tokenizer = pickle.load(open('model/tokenizers.pkl', 'rb'))
    le = pickle.load(open('model/le.pkl', 'rb'))
    model = keras.models.load_model('model/chat_model.h5')
    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def remove_punctuation(text):
    texts_p =[]
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

def predict(vector):
    output = model.predict(vector)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer

# def bag_of_words(s, words):
#     bag = [0 for _ in range(len(words))]
#     s_words = nltk.word_tokenize(s)
#     s_words = [stemmer.stem(word.lower()) for word in s_words]

#     for s_word in s_words:
#         for i, w in enumerate(words):
#             if w == s_word:
#                 bag[i] = 1

#     return np.array(bag)

# def chat():
    
#     while True:
#         inp = input("\n\nYou: ")
#         if inp.lower() == 'quit':
#             break

#     #Porbability of correct response 
#         results = model.predict([bag_of_words(inp, words)])

#     # Picking the greatest number from probability
#         results_index = np.argmax(results)

#         tag = labels[results_index]

#         for tg in data['intents']:

#             if tg['tag'] == tag:
#                 responses = tg['responses']
#                 print("Bot:\t" + random.choice(responses))
                
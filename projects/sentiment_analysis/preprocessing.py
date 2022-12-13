from nltk.tokenize import word_tokenize
import string
import numpy as np
import pandas as pd

def dataset_parsing():
    with open('sentiment_labelled_sentences/imdb_labelled.txt') as f:
        reviews = f.read()
        data = pd.DataFrame([review.split('\t') for review in reviews.split('\n')])
        data.columns = ['Review','Sentiment']
    
    return data


def split_word_reviews(data):
    text = list(data['Review'].values)
    clean_text = []
    for t in text:
        clean_text.append(t.translate(str.maketrans(
            '', '', string.punctuation)).lower().rstrip())

    tokenized = [word_tokenize(x) for x in clean_text]

    all_text = []
    for tokens in tokenized:
        for t in tokens:
            all_text.append(t)

    return tokenized, set(all_text)


def create_dictionaries(words):
    word_to_int_dict = {w: i+1 for i, w in enumerate(words)}
    int_to_word_dict = {i: w for w, i in word_to_int_dict.items()}
    return word_to_int_dict, int_to_word_dict

def pad_text(tokenized_reviews, seq_length):
    reviews = []
    for review in tokenized_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append(['']*(seq_length-len(review))+ review)
    
    return np.array(reviews)
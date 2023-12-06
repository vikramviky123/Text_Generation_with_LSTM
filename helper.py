import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import re
from typing import *

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

# fixed data
# Vocabulary size of the tokenizer
VOCAB_SIZE = 10000

# Maximum length of clean sentence is  217
# Giving 50 as the max length including the padded sequences
MAX_LENGTH = 225

# Output dimensions of the Embedding layer
EMBEDDING_DIM = 16

# Parameters for padding and OOV tokens
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"

# Clean the sentence


def clean_data(sentences: List[str]) -> List[str]:
    """
    Perform text cleaning on a list of sentences.

    Args:
        sentences (List[str]): List of input sentences.

    Returns:
        List[str]: List of cleaned sentences after applying stemming, lowercasing,
                   and removing stopwords.

    Note:
        This function uses the NLTK library for stemming and stopwords removal.
        Make sure to have NLTK installed (`pip install nltk`) and download the
        stopwords corpus using `nltk.download('stopwords')`.
    """
    ps = PorterStemmer()
    corpus = []

    for i in range(len(sentences)):
        each_sentence = re.sub('[^a-zA-Z]', ' ', sentences[i])
        each_sentence = each_sentence.lower()
        each_sentence = each_sentence.split()

        each_sentence = [ps.stem(word)
                         for word in each_sentence if word not in stopwords.words('english')]
        each_sentence = ' '.join(each_sentence)
        corpus.append(each_sentence)

    return corpus

###


def to_sequences_n_padding(tokenizer, MAX_LENGTH, PADDING_TYPE, TRUNC_TYPE, X_data):
    # Generate and pad the training sequences
    _sequences = tokenizer.texts_to_sequences(X_data)
    _padded = pad_sequences(_sequences,
                            maxlen=MAX_LENGTH,
                            padding=PADDING_TYPE,
                            truncating=TRUNC_TYPE)

    return _padded


def to_padding(sequences, MAX_LENGTH, PADDING_TYPE, TRUNC_TYPE):
    # Generate and pad the training sequences
    _padded = pad_sequences(sequences,
                            maxlen=MAX_LENGTH,
                            padding=PADDING_TYPE,
                            truncating=TRUNC_TYPE)

    return _padded


def to_sequences(tokenizer, X_data):
    # Generate and pad the training sequences
    _sequences = tokenizer.texts_to_sequences(X_data)

    return _sequences


def seq_gen(sent_seq):
    input_seq = []
    for line in sent_seq:
        for i in range(1, len(line)):
            input_seq.append(line[:i+1])

    return input_seq

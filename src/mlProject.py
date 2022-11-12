# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:41:35 2021

@author: dkina
"""
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import re
import nltk as nltk
from nltk.stem.snowball import SnowballStemmer
from keras.utils.generic_utils import get_custom_objects
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras import backend as K
from keras.layers import Dropout
from keras.layers import Conv1D
import tensorflow as tf
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.layers import Activation
from collections import Counter
import matplotlib.pyplot as plt


padding_len = 10000 #Used to pad the input word vector length
max_words = 5000 # This is the vocabulary size

# Loads the dataset file
def loadData():
    data = pd.read_csv("../data/ted_talks_en.csv")
    data = data.to_records(index=True)
    #data['topics'] = list(map(lambda x: x.replace('[','').replace(']','').replace('\'','').split(', '), data['topics']))
    #data = data[0:10]
    return data

# Analyzes the frequency of occurance of tags 
# and discards the non-frequent tags.
def analyzeTopics(tags):
    print("Analyzing tags...")
    allTags = []
    uniqueTags = {""}
    # Stores all the tags and convert it to array
    for i in range(len(tags)):
        tags[i] = tags[i].replace('\'','').replace('[','').replace(']','').split(', ')
        tags[i].append("TedX")
        for tag in tags[i]:
            allTags.append(tag)
            uniqueTags.add(tag)
    uniqueTags.remove("")

    # For every tag, get the frequency
    frequentTags = Counter(allTags).most_common(20)
    #print(frequentTags)
    #print(len(allTags))
    frequentTopics = [i[0] for i in frequentTags]
    # Remove the not-frequent tags
    i = 0
    for l in tags:
        l = [x for x in l if x in frequentTopics]
        tags[i]=l
        i=i+1
    # plot the tags frequency curve
    plt.plot([i[0] for i in frequentTags], [i[1] for i in frequentTags])
    plt.title("TED Topics Frequency distribution")
    plt.xticks(rotation=90)
    #print(tags)
    plt.show()
    return tags

# Transcript preprocessing.
# Stemming, Lemmatization
# Remove special characters and content inside paranthesis and quotes
# Remove Stop words and other irrelevant words. Lower case
# Tokenize the transcript  
def transcriptPreprocessing(data):
    print("Transcript Preprocessing...")
    scripts = data['transcript']
    i = 0
    j = 0
    for script in scripts:       
        #script = script.replace('(Applause)','').replace('(Laughter)','').replace('(Applause ends)','').replace('(Music)','').replace('(Music ends)','')
        #script = re.sub(r'[?|!|\'|"|#|.|,|\|/|\-|;|_|\!\[.*?\]|\!\(.*?\)]',r'',script)
        #script = re.sub(r'[!\[.*?\]|\!\(.*?\)|\!".*?"]',r'',script)
        # Remove content inside paranthesis
        #script = re.sub(r"\([^()]*\)|\[[^()]*\]|\"[^()]*\"", "", script)
        desc = data['description'][i]+" "+data['title'][i]
        
        # Keep only letters
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        script = ''.join(filter(whitelist.__contains__, script)).lower()
        desc = ''.join(filter(whitelist.__contains__, script)).lower()
        
        #script = script.lower()
        # Remove stop words and Stemming
        stop = set(stopwords.words("english"))
        stop.add('thank')
        stop.add('you')
        token_words=word_tokenize(script)
        
        #print(token_words)
        stemmer=SnowballStemmer("english", ignore_stopwords=True)
        lemmatizer = WordNetLemmatizer()
        stemmedScript = ""
        stemmedDesc = ""
        # Remove stop words and perform Lemmatization
        # Also combine the decription and title with the transcript
        # till it matches the padding length.
        # Remove this code if we do not want it.
        for word in token_words:
            word = word.strip()
            if word not in stop:
                stemmedScript = stemmedScript+" "+lemmatizer.lemmatize(word)
                #stemmedScript = stemmedScript+" "+stemmer.stem(word)
        token_words = word_tokenize(desc)
        # Remove stop words and perform Lemmatization
        for word in token_words:
            word = word.strip()
            if word not in stop:
                stemmedDesc = stemmedDesc+" "+lemmatizer.lemmatize(word)
        mergedScript = stemmedScript+" "+stemmedDesc
        while(len(mergedScript.split(" "))<padding_len):
            mergedScript = mergedScript + " " + stemmedScript+" "+stemmedDesc
        scripts[i] = mergedScript
        
        i=i+1
    data['transcript'] = scripts
    #print(data['transcript'][0])
    
    #Code to perform an EDA task on the transcript feature.
    # It calculates the words in each transcript.
    '''
    minlen = len(scripts[0])
    maxlen = len(scripts[0])
    for t in scripts:
        if len(t)<minlen:
            minlen = len(t)
        if len(t)>maxlen:
            maxlen = len(t)
    #print(j)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%",minlen)
    print(maxlen)
    '''
    return data

# Tokenize the transcript
def tokenizeTranscript(data):
    X = data['transcript']
    # The number of unique words to consider.
    # The higher frequency words from the corpus will be used.
    # In total there are about 50000 unique words including all ted talks
    
    # The input to the first layer of the DNN will be tokenized list of words.
    # This should be constant for all the ted talks.
    # This is the limit of word counts in each ted talk.
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)
    
    tokenizer.fit_on_texts(X)
    #print(tokenizer.word_index)
    
    seq = tokenizer.texts_to_sequences(X)
    features = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=padding_len, padding='post', value=0)
    #print(features.shape)
    #print(features[1])
    return features, tokenizer

# The NN model is constructed here.
def deepNeural(X, y, t):
    
    #print(len(X[0]))
    print("Building the Neural Network Model...")
    
    # GloVe Embedding
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('../data/glove.6B.100d.txt', encoding="utf8")
    for line in f:   
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((len(t.word_index) + 1, 100))
    for word, i in t.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    print("GloVe embedding is done successfully...")
    # print(embedding_matrix.shape)
    # print(len(t.word_index) + 1)
   
    # Defining the neural network model layes
    model = Sequential()
    model.add(Embedding(len(t.word_index) + 1, 100, weights=[embedding_matrix], input_length=len(X[0]), trainable=False))
    #model.add(Embedding(max_words, 50, input_length=len(X[0])))
    #model.add(Flatten())
    #model.add(Dropout(0.1))
    model.add(SimpleRNN(50,activation='tanh', use_bias=True))
    
    #model.add(Conv1D(300, 2, padding='valid', activation='relu', strides=1))
    #model.add(GlobalMaxPool1D())
    # model.add(Dropout(0.1))
    # model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
    #model.add(Dense(len(y[0]), kernel_initializer='he_uniform', activation='sigmoid'))
    model.add(Dense(len(y[0]), activation='sigmoid', use_bias=True))
    
    print(model.summary())
    
    #Changing the learning rate to 0.1
    opt = Adam(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['acc',f1_m,precision_m, recall_m])
    
    # Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X=np.asarray(X).astype(np.float32)
    #X=np.asarray(X).astype(np.float32)
    #X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.1, random_state = 48)
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.3)
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test)
   
    # Predict accuracy of the test dataset
    xt = model.predict(X_test)
    by = model.layers[2].get_weights()
    
    #print(by)
    print("predicted weight verctor = ", xt[1])
    print("actual label = ", y_test[1])
    
    print("loss = "+ str(loss)+"\naccuracy = "+  str(accuracy*100)+"\nf1_score =  "+ str(f1_score)+" \nprecision = "+  str(precision)+" \nrecall = "+  str(recall))
    #print(history)
    return model

# Custom functions for 
# Precision, Recall, F1 scored (Macro)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def flow():
    
    data = loadData()
    
    tags = analyzeTopics(data['topics'])
    #print(data['topics'])
    data['topics']=tags # Store the new tags
    data = transcriptPreprocessing(data) # Focus on the Transcrips
    
    X, tokenizer = tokenizeTranscript(data)
    df = pd.DataFrame(data=X)
    df.to_csv('../data/Features.csv', sep=',', header=False, float_format='%.2f', index=False)
    
    mlb = MultiLabelBinarizer()
    
    mlb.fit(data['topics'])
    
    y = mlb.transform(data['topics'])
    df = pd.DataFrame(data=y)
    df.to_csv('../data/MultiLabels.csv', sep=',', header=False, float_format='%.2f', index=False)
    print("Top frequency labels filtered for multi-labeling -")
    print(mlb.classes_)
    
    X = open("../data/Features.csv")
    X = np.loadtxt(X, delimiter=",")
    y = open("../data/MultiLabels.csv")
    y = np. loadtxt(y, delimiter=",")
    # y = pd.read_csv("../data/MultiLabels.csv")
    # y = y.to_records(index=True)
    print(y.shape, "sdfa",X.shape)
    #print(X[0:2])
    
    
    model = deepNeural(X, y, tokenizer)
    
    #train(model, X, y)
    #print(data['transcript'])
    #p = model.predict(data[3])
    #print(p)
    #print(tags)
    
flow()


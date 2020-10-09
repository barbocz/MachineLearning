#  https://colab.research.google.com/drive/1ysEKrw_LE2jMndo1snrZUh5w87LQsCxk#forceEdit=true&sandboxMode=true&scrollTo=pdsus1kyXWC8
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import pandas as pd
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64


with np.load('imdb.npz',allow_pickle=True) as f:
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']

print(x_train[1])

# imdb_data_df=pd.read_csv('IMDB Dataset.csv')
# print(imdb_data_df.head(1))
# imdb_data=imdb_data_df.to_numpy()
#
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

start_time = time.time()

#%% load dataset
dataset = pd.read_csv('questions.csv')

#%% load question pairs
q1 = dataset['question1'].values
q2 = dataset['question2'].values
isDuplicate = dataset['is_duplicate'].values
print (len(q1), "Question Pairs Loaded")

## CONSTANTS
TRAINING_SIZE = int (len(q1) * 0.95)
VALIIDATION_SIZE = int(len(q1) * 0.025)
TEST_SIZE = int(len(q1) * 0.025)
# max sentene length
MAX_LENGTH = 30

## Tokenize data
docs = np.append(q1, q2)
t = Tokenizer(oov_token='UNK')
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print ('Vocab size:', vocab_size)

## Text to integers
encoded_q1 = t.texts_to_sequences(q1)
encoded_q2 = t.texts_to_sequences(q2)
padded_q1 = pad_sequences(encoded_q1, maxlen=MAX_LENGTH, padding='post')
padded_q2 = pad_sequences(encoded_q2, maxlen=MAX_LENGTH, padding='post')

#%% Training Data
train_x1 = padded_q1[:TRAINING_SIZE]
train_x2 = padded_q2[:TRAINING_SIZE]
train_y = isDuplicate[:TRAINING_SIZE]
print ('Generated', len(train_x1), 'Training Set')
print (train_x1[0])

#%% Validation Data
valid_x1 = padded_q1[TRAINING_SIZE:TRAINING_SIZE+VALIIDATION_SIZE]
valid_x2 = padded_q2[TRAINING_SIZE:TRAINING_SIZE+VALIIDATION_SIZE]
valid_y = isDuplicate[TRAINING_SIZE:TRAINING_SIZE+VALIIDATION_SIZE]
print ('Generated',len(valid_x1) , 'Validation Set')


## Test Data
test_x1 = padded_q1[-TEST_SIZE:]
test_x2 = padded_q2[-TEST_SIZE:]
test_y = isDuplicate[-TEST_SIZE:]
print ('Generated',len(test_x1) , 'Test Set')

## Save Data
np.save('datasets/train_x1', train_x1)
np.save('datasets/train_x2', train_x2)
np.save('datasets/train_y', train_y)
np.save('datasets/valid_x1', valid_x1)
np.save('datasets/valid_x2', valid_x2)
np.save('datasets/valid_y', valid_y)
np.save('datasets/test_x1', test_x1)
np.save('datasets/test_x2', test_x2)
np.save('datasets/test_y', test_y)
print ('Datasets saved')


## Load Word Embeddings
word_index = t.word_index
GLOVE_FILE = 'glove.840B.300D.txt'
embeddings_index = {}
with open(GLOVE_FILE, encoding='utf-8') as f:
	l = 0
	for line in f:
		values = line.split(' ')
		word = values[0]
		embedding = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = embedding
		print (l)
		l += 1
	embeddings_index['UNK'] = np.zeros(300)

print('Word embeddings: %d' % len(embeddings_index))

nb_words = len(word_index)
word_embedding_matrix = np.zeros((nb_words + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

print ('Created Embedding Matrix')
np.save('embedding_matrix', word_embedding_matrix)

print ('Total time:', time.time() - start_time)

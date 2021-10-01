# Almost done with model creation
# Remaining task is to retrieve a larger data set and train it again properly

# Then, save it in proper format (tensorflow lite file)
# At last, use this file in the android application
# and complete the project

import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

# !wget https://www.gutenberg.org/files/1661/1661-0.txt

path = '1661-0.txt'
text = open(path).read().lower()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
  prev_words.append(words[i:i + WORD_LENGTH])
  next_words.append(words[i+WORD_LENGTH])

X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)

for i, each_words in enumerate(prev_words):
  for j, each_word in enumerate(each_words):
    X[i, j, unique_word_index[each_word]] = 1
  	Y[i, unique_word_index[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))

model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

def prepare_input(text):
  x = np.zeros((1, WORD_LENGTH, len(unique_words)))
  for t, word in enumerate(text.split()):
    print(word)
    x[0, t, unique_word_index[word]] = 1
  return x

prepare_input("It is not a lack".lower())

def sample(preds, top_n=3):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds)

  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)

  return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completions(text, n=3):
  if text == "":
    return("0")
  x = prepare_input(text)
  preds = model.predict(x, verbose=0)[0]
  next_indices = sample(preds, n)
  return [unique_words[idx] for idx in next_indices]

q = "Your life will never be the same again"
print("correct sentence: ", q)

seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence: ", seq)
print("Next possible words: ", predict_completions(seq, 5))


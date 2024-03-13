import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import re
def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:
            long_words.append(i)
    return (" ".join(long_words)).strip()
# Open the file and read its contents
with open('/content/lyrics_dataset.txt', 'r') as file:
    data_text = file.read()


# preprocess the text
data_new = text_cleaner(data_text)
def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

# create sequences
sequences = create_seq(data_new)
# create a character mapping index
chars = sorted(list(set(data_new)))
mapping = dict((c, i) for i, c in enumerate(chars))

def encode_seq(seq):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences

# encode the sequences
sequences = encode_seq(sequences)
from sklearn.model_selection import train_test_split

# vocabulary size
vocab = len(mapping)
sequences = np.array(sequences)
# create X and y
X, y = sequences[:,:-1], sequences[:,-1]
# one hot encode y
y = to_categorical(y, num_classes=vocab)
# create train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)
# define model
model = Sequential()
model.add(Embedding(vocab, 50, input_length=30, trainable=True))
model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab, activation='softmax'))
print(model.summary())
# compile the model
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# fit the model
model.fit(X_tr, y_tr, epochs=10, verbose=2, validation_data=(X_val, y_val))
import numpy as np

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict character
        yhat = model.predict(encoded, verbose=0)
        # map predicted integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == np.argmax(yhat):
                out_char = char
                break
        # append to input
        in_text += out_char
    return in_text
inp = " "
print(len(inp))
print(generate_seq(model,mapping,30,inp.lower(),15))
from math import exp
from keras.preprocessing.sequence import pad_sequences

# Calculate Perplexity
def calculate_perplexity(model, X_val, y_val):
    # Evaluate the model on validation set
    loss = model.evaluate(X_val, y_val, verbose=0)
    cross_entropy = loss[0]
    # Calculate perplexity
    perplexity = exp(cross_entropy)
    return perplexity

# Generate Text Samples
def generate_text_samples(model, mapping, seq_length, seed_texts, n_chars):
    generated_texts = []
    for seed_text in seed_texts:
        generated_text = generate_seq(model, mapping, seq_length, seed_text.lower(), n_chars)
        generated_texts.append(generated_text)
    return generated_texts

# Calculate perplexity
perplexity = calculate_perplexity(model, X_val, y_val)
print("Perplexity:", perplexity)

# Generate text samples
seed_texts = ["hello", "the cat", "machine learning"]
generated_texts = generate_text_samples(model, mapping, 30, seed_texts, 100)
for i, text in enumerate(generated_texts):
    print(f"Generated Text {i+1}: {text}")
model.save('text quality.h5')

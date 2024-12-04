import random
from collections import Counter
import pandas as pd
import os
import string
import nltk
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
os.chdir(r'C:\Users\Klaws\Downloads\UWL Fall 2024\ML_Crypto Research')

#%%

# Text cleaning function
def remove_non_letters(text):
    text = text.lower()
    return re.sub(r'[^A-Za-z\s]', '', text)

# Generate random substitution key
def generate_sub_key():
    alphabet = list(string.ascii_uppercase)
    shuffled = alphabet[:]
    random.shuffle(shuffled)
    return ''.join(shuffled)

# Substitution encryption
def sub_encrypt(plaintext, key):
    plaintext = plaintext.upper()
    key_map = {original: substitute for original, substitute in zip(string.ascii_uppercase, key)}
    ciphertext = ''.join(key_map[char] if char in key_map else char for char in plaintext)
    return ciphertext

# Substitution Cipher Decryption
def sub_decrypt(ciphertext, key):
    ciphertext = ciphertext.upper()
    reverse_key_map = {substitute: original for original, substitute in zip(string.ascii_uppercase, key)}
    plaintext = ''.join(reverse_key_map[char] if char in reverse_key_map else char for char in ciphertext)
    return plaintext

# Letter Frequencies
def char_frequency(text):
    text = ''.join(filter(str.isalpha, text))  
    counts = Counter(text)  
    total_chars = len(text)  
    return {char: count / total_chars for char, count in counts.items()}

def trigram_frequency(text):
    trigrams = [text[i:i+3] for i in range(len(text)-2)]  # Ensure trigrams are of length 3
    total_trigrams = len(trigrams)
    trigram_counts = Counter(trigrams)  # Count occurrences of each trigram
    return {trigram: count / total_trigrams for trigram, count in trigram_counts.items()}

# Split the text into chunks of 100 words
def split_text_into_chunks_by_words(text, chunk_size=100):
    words = nltk.word_tokenize(text)  # Tokenize into words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# %% Read Text File 
with open('SOTL.txt', 'r', encoding='utf-8') as file:
    chapter_text = file.read()

chunks = split_text_into_chunks_by_words(chapter_text, chunk_size=100)

#%% Encrypt and store sentences for Substitution Cipher with random keys
plain_cipher_pairs = []
for chunk in chunks:
    raw_plaintext = remove_non_letters(chunk)  # Clean the plaintext
    key = generate_sub_key()  # Generate a random substitution key
    ciphertext = sub_encrypt(raw_plaintext, key)  # Encrypt the plaintext
    plain_cipher_pairs.append((key, raw_plaintext, ciphertext))  # Store key, plaintext, and ciphertext

# Create a DataFrame to store results
df = pd.DataFrame(plain_cipher_pairs, columns=['Key', 'Plaintext', 'Ciphertext'])

#%% Feature extraction for ML model

# Create feature set for ML training
df_ml_dataset = df.copy()
df_ml_dataset['Ciphertext_Trigrams'] = df_ml_dataset['Ciphertext'].apply(trigram_frequency)
df_ml_dataset = pd.DataFrame(df_ml_dataset['Ciphertext_Trigrams'].to_list()).fillna(0)
df_ml_dataset = df_ml_dataset[sorted(df_ml_dataset.columns)]
df_ml_dataset = pd.concat([df[['Key', 'Plaintext', 'Ciphertext']], df_ml_dataset.add_prefix('Cipher_')], axis=1)

# %% Prepare data for training

# Features: Frequency distribution of ciphertext letters
X = np.array(df_ml_dataset.drop(columns=['Key', 'Plaintext', 'Ciphertext']))

# Target: Binary encoding for keys (0 or 1)
# For example, classify key as either 0 or 1 based on the first letter of the key
# Change this logic to fit your exact binary classification scheme
y = (df_ml_dataset['Key'].apply(lambda x: 0 if string.ascii_uppercase.index(x[0]) % 2 == 0 else 1)).values  # Example binary logic

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Build and compile the neural network model

model = Sequential()


model.add(Dense(676, input_dim=X.shape[1], activation='sigmoid'))

# Output layer: 1 neuron with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model using binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% Train the model

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# %% Evaluate the model

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

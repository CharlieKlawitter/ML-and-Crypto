import random
from collections import Counter
import pandas as pd
import os
import nltk
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
os.chdir(r'C:\Users\Klaws\Downloads\UWL Fall 2024\ML_Crypto Research')


# %% Shift Cipher Functions

# text cleaning function to remove non-letter characters
def remove_non_letters(text):
    return re.sub(r'[^A-Za-z\s]', '', text)  # Only keep letters and spaces

# Encryption using shift cipher
def shift_cipher(plaintext, key):
    ciphertext = ""
    plaintext = remove_non_letters(plaintext)
    plaintext = plaintext.replace(" ", "")
    plaintext = plaintext.lower()
    for char in plaintext:
        ciphertext += chr((ord(char) + key - 97) % 26 + 97)  # Shift the letter
        
    ciphertext = ciphertext.upper()  # Convert to uppercase

    return plaintext, ciphertext

# Decryption using the inverse shift
def decrypt_with_shift(ciphertext, shift):
    decrypted_text = []
    for char in ciphertext:
        if 'A' <= char <= 'Z':  # Check if char is an uppercase letter
            decrypted_char = chr(((ord(char) - ord('A') - shift) % 26) + ord('A'))
            decrypted_text.append(decrypted_char)
    return ''.join(decrypted_text).lower()

# Letter frequency counting function
def char_frequency(text):
    counts = Counter(text)
    total_chars = len(text)
    return {char: count / total_chars for char, count in counts.items()}

# %% Text Preprocessing

# Split the text into chunks of 100 words
def split_text_into_chunks_by_words(text, chunk_size=100):
    words = nltk.word_tokenize(text)  # Tokenize into words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# %% Read and process the plaintext

with open('SOTL.txt', 'r', encoding='utf-8') as file:
    chapter_text = file.read()

# Split the text into chunks of 100 words
chunks = split_text_into_chunks_by_words(chapter_text, chunk_size=100)

# %% Encrypt each chunk and store the result

plain_cipher_pairs = []
for chunk in chunks:
    key = random.randint(1, 25)  # Random key for shift cipher
    plaintext, ciphertext = shift_cipher(chunk, key)
    plain_cipher_pairs.append((key, plaintext, ciphertext))

df = pd.DataFrame(plain_cipher_pairs, columns=['Key', 'Plaintext', 'Ciphertext'])

# %% Feature extraction for ML model

# Create feature set for ML training
df_ml_dataset = df.copy()
df_ml_dataset['Ciphertext_Freq'] = df_ml_dataset['Ciphertext'].apply(char_frequency)
df_ml_dataset = pd.DataFrame(df_ml_dataset['Ciphertext_Freq'].to_list()).fillna(0)
df_ml_dataset = df_ml_dataset[sorted(df_ml_dataset.columns)]
df_ml_dataset = pd.concat([df[['Key', 'Plaintext', 'Ciphertext']], df_ml_dataset.add_prefix('Cipher_')], axis=1)

# %% Prepare data for training

# Features: Frequency distribution of ciphertext letters
X = np.array(df_ml_dataset.drop(columns=['Key', 'Plaintext', 'Ciphertext']))

# Target: One-hot encoded labels for the key (0-25)
y = to_categorical(df_ml_dataset['Key'], num_classes=26)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Build and compile the neural network model

model = Sequential()

# Hidden layer: 26 neurons, input dimension 26 (ciphertext frequencies)
model.add(Dense(26, input_dim=26, activation='sigmoid'))  

# Output layer: 26 neurons, one for each possible key (0-25)
model.add(Dense(26, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# %% Train the model 

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# %% Evaluate the model

loss, accuracy = model.evaluate(X_train, y_train)
print(f"Model Loss: {loss}")
print(f"Model Accuracy: {accuracy}")











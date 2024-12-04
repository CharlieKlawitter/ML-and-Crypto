import random as random
import string
from collections import Counter
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
import os
os.chdir(r'C:\Users\Klaws\Downloads\UWL Fall 2024\ML_Crypto Research')
import nltk
nltk.download('punkt') # text parser
import re

#%%

# Text cleaning function
def remove_non_letters(text):
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
    counts = Counter(text)
    total_chars = len(text)
    return {char: count / total_chars for char, count in counts.items()}

def score_text(text):
    english_letter_freq = {
       'E': 12.70, 'T': 9.10, 'A': 8.20, 'O': 7.50, 'I': 7.00, 'N': 6.70, 
       'S': 6.30, 'H': 6.10, 'R': 6.00, 'D': 4.30, 'L': 4.00, 'C': 2.80, 
       'U': 2.80, 'M': 2.40, 'W': 2.30, 'F': 2.20, 'G': 2.00, 'Y': 2.00, 
       'P': 1.90, 'B': 1.50, 'V': 1.00, 'K': 0.80, 'J': 0.20, 'X': 0.10, 
       'Q': 0.10, 'Z': 0.10
    }
    text_counter = Counter(text.upper())
    text_len = len(text)
    score = 0
    for letter, count in text_counter.items():
        if letter in english_letter_freq:
            score += (count / text_len) * english_letter_freq[letter]
    return score 

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

# Decrypt the ciphertext using the generated key and validate
df['Decrypted_Plaintext'] = df.apply(lambda row: sub_decrypt(row['Ciphertext'], row['Key']), axis=1)
df['Matches'] = df['Plaintext'] == df['Decrypted_Plaintext']

# Summary of results
false_count = len(df) - df['Matches'].sum()
print(f"CRYPTANALYSIS: Number of non-matching decryptions: {false_count}")
accuracy = (len(df) - false_count) / len(df)
print(f"CRYPTANALYSIS: Accuracy of decryption with known keys: {accuracy:.4f}")

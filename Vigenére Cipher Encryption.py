import random as random
import string
from collections import Counter, defaultdict
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

# Vigenère Cipher Encryption
def vig_encrypt(plaintext, key):
    ciphertext = ""
    plaintext = remove_non_letters(plaintext).replace(" ", "").lower()
    key = remove_non_letters(key).upper()

    key_length = len(key)
    key_index = 0
    
    for char in plaintext:
        if 'a' <= char <= 'z':
            shift = ord(key[key_index % key_length]) - ord('A')
            ciphertext += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            key_index += 1

    return plaintext, ciphertext.upper()

# Vigenère Cipher Decryption
def vig_decrypt(ciphertext, key):
    decrypted_text = ""
    key = remove_non_letters(key).upper()
    key_length = len(key)
    key_index = 0
    
    for char in ciphertext:
        if 'A' <= char <= 'Z':
            shift = ord(key[key_index % key_length]) - ord('A')
            decrypted_text += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            key_index += 1

    return decrypted_text.lower()

# Random Key Generator
def random_key(min_length=3, max_length=20):
    key_length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(key_length))

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

# Kasiski Examination (find repeated trigrams and their distances)
def kasiski_examination(ciphertext):
    seq_distances = defaultdict(list)
    for i in range(len(ciphertext) - 3):
        trigram = ciphertext[i:i+3]
        for j in range(i + 3, len(ciphertext) - 3):
            if ciphertext[j:j+3] == trigram:
                seq_distances[trigram].append(j - i)
    
    return seq_distances

def get_factors(n):
    factors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:  # Avoid adding the square root twice if it's a perfect square
                factors.append(n // i)
    return factors

# Estimate key length based on Kasiski
def estimate_key_length(ciphertext):
    distances = kasiski_examination(ciphertext)
    
    if not distances:
        return 3  # Fallback value if no trigrams are found
    
    factors = []
    for trigram, positions in distances.items():
        for dist in positions:
            factors.extend(get_factors(dist))
    
    if not factors:
        return 3  # Fallback if no factors are found
    
    factor_counts = Counter(factors)
    return factor_counts.most_common(1)[0][0]  # Return the most common factor (estimated key length)

# Improved Subcipher splitting to handle edge cases
def break_into_subciphers(ciphertext, key_length):
    if key_length <= 0:
        raise ValueError("Key length must be greater than 0.")
    
    subciphers = ['' for _ in range(key_length)]
    for i, char in enumerate(ciphertext):
        subciphers[i % key_length] += char
    return subciphers

# Frequency analysis to determine shift
def caesar_shift_decrypt(ciphertext, shift):
    return ''.join([chr((ord(c) - ord('A') - shift) % 26 + ord('A')) for c in ciphertext])

def frequency_analysis(subcipher):
    english_frequencies = {'A': 8.167, 'B': 1.492, 'C': 2.782, 'D': 4.253, 'E': 12.702, 'F': 2.228, 'G': 2.015, 'H': 6.094, 
                           'I': 6.966, 'J': 0.153, 'K': 0.772, 'L': 4.025, 'M': 2.406, 'N': 6.749, 'O': 7.507, 'P': 1.929, 
                           'Q': 0.095, 'R': 5.987, 'S': 6.327, 'T': 9.056, 'U': 2.758, 'V': 0.978, 'W': 2.360, 'X': 0.150, 
                           'Y': 1.974, 'Z': 0.074}
    
    best_shift = 0
    max_score = float('-inf')
    for shift in range(26):
        decrypted = caesar_shift_decrypt(subcipher, shift)
        score = sum([decrypted.count(c) * english_frequencies.get(c, 0) for c in decrypted])
        if score > max_score:
            max_score = score
            best_shift = shift
    
    return best_shift

# Decrypt using Vigenère with estimated key length
def decrypt_vigenere(ciphertext, key_length):
    subciphers = break_into_subciphers(ciphertext, key_length)
    key = ''
    for subcipher in subciphers:
        shift = frequency_analysis(subcipher)
        key += chr(shift + ord('A'))
    
    # Decrypt the ciphertext using the estimated key
    decrypted_text = vig_decrypt(ciphertext, key)
    
    return decrypted_text, key  # Return both the plaintext and the estimated key

# %% Read Text File 
with open('SOTL.txt', 'r', encoding='utf-8') as file:
    chapter_text = file.read()
    
sentences = nltk.tokenize.sent_tokenize(chapter_text)

# %% Encrypt and store sentences for Vigenère Cipher with random keys 
plain_cipher_pairs = []
for sentence in sentences:
    raw_plaintext = sentence 
    key = random_key(3,20)
    plaintext, ciphertext = vig_encrypt(raw_plaintext, key)
    plain_cipher_pairs.append((key, plaintext, ciphertext))
    
df = pd.DataFrame(plain_cipher_pairs, columns=['Key', 'Plaintext', 'Ciphertext'])

# %% Cryptanalysis by comparison to english letter frequency
df_cryptanalysis = df.copy()
df_cryptanalysis['best_key'] = None
df_cryptanalysis['best_plaintext'] = None

for i in range(df_cryptanalysis.shape[0]):
    ciphertext = df_cryptanalysis.loc[i, 'Ciphertext']
    key_length = estimate_key_length(ciphertext)
    
    best_plaintext, best_key = decrypt_vigenere(ciphertext, key_length)  # Get both plaintext and key
    
    df_cryptanalysis.loc[i, 'best_key'] = best_key  # Store the estimated key
    df_cryptanalysis.loc[i, 'best_plaintext'] = best_plaintext  # Store the estimated plaintext

# Check if the actual and estimated keys match
df_cryptanalysis['keys_match'] = df_cryptanalysis['Key'] == df_cryptanalysis['best_key']

# Summary of results
false_count = len(df_cryptanalysis) - df_cryptanalysis['keys_match'].sum()
print(f"CRYPTANALYSIS: Number of non-matching keys: {false_count}")
acc_perc = (len(df_cryptanalysis)-false_count)/len(df_cryptanalysis)
print(f"CRYPTANALYSIS: Percent of matching keys: {acc_perc:.4f}")
























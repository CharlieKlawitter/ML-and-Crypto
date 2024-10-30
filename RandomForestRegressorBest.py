import random as random
from collections import Counter
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
import os
os.chdir(r'C:\Users\Klaws\Downloads\Coding\UpdatedMLCrypto')
import nltk
nltk.download('punkt') # text parser
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
# %% custom functions

# text cleaning
def remove_non_letters(text):
    return re.sub(r'[^A-Za-z\s]', '', text)  # Only keep letters and spaces


# encryption via shift
def shift_cypher(plaintext, key):
    ciphertext = ""
    plaintext = remove_non_letters(plaintext)
    plaintext = plaintext.replace(" ", "")
    plaintext = plaintext.lower()
    for char in plaintext:
        ciphertext += chr((ord(char) + key - 97) % 26 + 97)
        
    ciphertext = ciphertext.upper()

    return plaintext, ciphertext 


# decryption via unshift
def decrypt_with_shift(ciphertext, shift):
    decrypted_text = []
    for char in ciphertext:
        if 'A' <= char <= 'Z':  # Check if char is an uppercase letter
            decrypted_char = chr(((ord(char) - ord('A') - shift) % 26) + ord('A'))
            decrypted_text.append(decrypted_char)
    return ''.join(decrypted_text).lower()


# letter frequency counts
def char_frequency(text):
    counts = Counter(text)
    total_chars = len(text)
    return {char: count / total_chars for char, count in counts.items()}


# score text based on letter frequencies
def score_text(text):
    english_letter_freq = {
        'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 
        'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 
        'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 
        'P': 1.93, 'B': 1.49, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 
        'Q': 0.10, 'Z': 0.07
    }
    text_counter = Counter(text.upper())
    text_len = len(text)
    score = 0 # similar to dot product (cosine similarity) in for loop calc
    for letter, count in text_counter.items():
        if letter in english_letter_freq:
            score += (count / text_len) * english_letter_freq[letter]
    return score


# break shift cypher
def crack_shift_cipher_auto(ciphertext):    
    # Try all possible shifts and score the resulting plaintexts
    best_shift = 0
    best_score = -1
    best_plaintext = ""
    
    for shift in range(26):
        decrypted = decrypt_with_shift(ciphertext, shift)
        current_score = score_text(decrypted)
        if current_score > best_score:
            best_score = current_score
            best_shift = shift
            best_plaintext = decrypted
    
    return best_shift, best_plaintext


# %% read in text file

with open('lotr.txt', 'r', encoding='utf-8') as file:
    chapter_text = file.read()

sentences = nltk.tokenize.sent_tokenize(chapter_text)

# %% encrypt and store sentences

plain_cypher_pairs = []
for sentence in sentences:
    raw_plaintext = sentence
    key = random.randint(1,25)
    plaintext, ciphertext = shift_cypher(raw_plaintext, key)
    plain_cypher_pairs.append((key, plaintext, ciphertext))

df = pd.DataFrame(plain_cypher_pairs, columns=['Key', 'Plaintext', 'Ciphertext'])

# %% full features for ML model training

df_ml_dataset = df.copy()
df_ml_dataset['Ciphertext_Freq'] = df_ml_dataset['Ciphertext'].apply(char_frequency)
df_ml_dataset = pd.DataFrame(df_ml_dataset['Ciphertext_Freq'].to_list()).fillna(0)
df_ml_dataset = df_ml_dataset[sorted(df_ml_dataset.columns)]
df_ml_dataset = pd.concat([df[['Key', 'Plaintext', 'Ciphertext']], df_ml_dataset.add_prefix('Cipher_')], axis=1)

# %% cryptanalysis by comparison to english letter frequency

df_cryptanalysis = df.copy()

df_cryptanalysis['best_key'] = None
df_cryptanalysis['best_plaintext'] = None

for i in range(df_cryptanalysis.shape[0]):
    ciphertext = df_cryptanalysis.loc[i, 'Ciphertext']
    best_key, best_plaintext = crack_shift_cipher_auto(ciphertext)
    
    df_cryptanalysis.loc[i, 'best_key'] = best_key
    df_cryptanalysis.loc[i, 'best_plaintext'] = best_plaintext
    
df_cryptanalysis['best_key'] = pd.to_numeric(df_cryptanalysis['best_key'], errors='coerce')
    
df_cryptanalysis['keys_match'] = df_cryptanalysis['Key'] == df_cryptanalysis['best_key']

false_count = len(df_cryptanalysis) - df_cryptanalysis['keys_match'].sum()
print(f"CRYPTANALYSIS: Number of non-matching keys: {false_count}")
acc_perc = (len(df_cryptanalysis)-false_count)/len(df_cryptanalysis)
print(f"CRYPTANALYSIS: Percent of matching keys: {acc_perc:.4f}")

# %% ML model train-test

df_ml_results = df_ml_dataset.copy()

X = df_ml_results.iloc[:, 3:]
y = df_ml_results['Key']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% 
#Randomized search 

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth':  [None, 10, 20],
    'min_samples_split':  [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
    




model = RandomForestClassifier(n_estimators=300)
search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
search.fit(X_train, y_train)

best_model= search.best_estimator_

train_accuracy = best_model.score(X_train, y_train)
print(f"ML Model train accuracy: {train_accuracy:.4f}")
accuracy = best_model.score(X_test, y_test)
print(f"ML Model test accuracy: {accuracy:.4f}")

train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

mae_train = mean_absolute_error(y_train, train_preds)
mae_test = mean_absolute_error(y_test, test_preds)


df_ml_results.loc[X_train.index, 'InSample_Predictions'] = train_preds  # In-sample predictions
df_ml_results.loc[X_test.index, 'OutSample_Predictions'] = test_preds   # Out-of-sample predictions

df_ml_results.loc[X_train.index, 'InSample_Correct'] = (np.round(train_preds) == y_train)   # True/False for in-sample
df_ml_results.loc[X_test.index, 'OutSample_Correct'] = (np.round(test_preds) == y_test)    # True/False for out-of-sample

df_ml_results = df_ml_results.drop(df_ml_results.filter(like="Cipher_").columns, axis=1)


# %% ML model k-fold cross validation

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=300)
scores = cross_val_score(model, X, y, cv=kf)

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))

# %%

plt.scatter(y_test, test_preds)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.show()
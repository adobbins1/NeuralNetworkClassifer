# Austin Dobbins
# DSC550-T301
# Exercise 9.2

import numpy as np, json, re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


def clean_text(text):
    text = text.lower()
    text = text.lower()
    text = re.sub('&lt;/?.*?&gt;', ' &lt;&gt', text)
    text = re.sub('\\d|\\W+|_', ' ', text)
    text = re.sub('[^a-zA-Z]', " ", text)

    return text


# Create stop words list
stop_words = stopwords.words('english')

coms = pd.read_json(r'C:\Users\austi\OneDrive\Desktop\categorized-comments.jsonl', lines=True)

# Details of JSON
print('Size: ', len(coms), '\n',
      'Shape: ', coms.info(), '\n',
      'Categories: ', coms['cat'].unique())

# Cutting Size of JSON for Performance
size = 2500
replace = True
fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace), :]

cons = coms.groupby("cat", as_index=False).apply(fn)

# Details of JSON after Cutting Size
print('Size: ', len(cons), '\n',
      'Shape: ', cons.info(), '\n',
      'Categories: ', cons['cat'].unique())

# Creating Feature Matrix, Target and Sample, and Train Test Split
cv = CountVectorizer(stop_words=stop_words)
x = cv.fit_transform((cons['txt']))
y = cons['cat']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=75)
mlp.fit(X_train, y_train)

# Calculating Accuracy of Model
predictions = mlp.predict(X_test)
print('Confusion Matrix: ', confusion_matrix(y_test, predictions))
print('Classification Report:', classification_report(y_test, predictions))
print('Accuracy: ', accuracy_score(y_test, predictions))


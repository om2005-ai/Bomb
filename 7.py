import nltk
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = """Albert Einstein was born in Ulm, Germany in 1879.
He was a great scientist and developed the theory of relativity."""

from nltk.tokenize import sent_tokenize, word_tokenize
sentences = sent_tokenize(text)
words = word_tokenize(text)
print("Sentences:", sentences)
print("Words:", words)

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
filtered_words = [w for w in words if w.lower() not in stop_words]
print("Filtered Words:", filtered_words)

from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered_words]
print("Stemmed Words:", stemmed_words)

from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
lemmatized_words = [lem.lemmatize(w) for w in filtered_words]
print("Lemmatized Words:", lemmatized_words)

import nltk
nltk.download('averaged_perceptron_tagger_eng')

import nltk
pos_tags = nltk.pos_tag(words)
print("POS Tags:", pos_tags)

from sklearn.feature_extraction.text import TfidfVectorizer
documents = [text]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)
print("TF-IDF Matrix:\n", X.toarray())
print("Feature Names:\n", tfidf.get_feature_names_out())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

tf = TfidfVectorizer()
X = tf.fit_transform(data['Phrase'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

clf = MultinomialNB()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
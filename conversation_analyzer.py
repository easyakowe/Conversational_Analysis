""" Model to summarize what a conversation is centered on by a particular user """
import re
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# import pandas as pd


def starts_with_date(sttrng):
    """ Regular expression to recognize a text structure that starts with a date """
    pattern = 'REGEX'
    result = re.match(pattern, sttrng)
    if result:
        return True
    return False


def starts_with_author(sttrng):
    """ Regular expression to recognize a text structure that starts with an author """
    patterns = ['REGEX']
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, sttrng)
    if result:
        return True
    return False


def get_data_point(line):
    """ Function to read the line and convert to valid text """

    split_line = line.split(' - ')

    date_time = split_line[0] # dateTime = '18/06/17, 22:47'

    date, time = date_time.split(', ') # date = '18/06/17'; time = '22:47'

    message = ' '.join(split_line[1:])

    if starts_with_author(message):
        split_message = message.split(': ')
        author = split_message[0] # author = 'Nano'
        message = ' '.join(split_message[1:]) # message = 'Why do you have 2 numbers, Pat?'
    else:
        author = None
    return date, time, author, message

print(get_data_point('18/06/17, 22:47 - Nano: Why do you have 2 numbers, Pat?'))


CHAT_TRAIN = []
for line2 in open('test_chat.txt', 'r'):
    CHAT_TRAIN.append(get_data_point(line2)[3])

CHAT_TEST = []
for line1 in open('train_chat.txt', 'r'):
    CHAT_TEST.append(get_data_point(line1)[3])

len(CHAT_TRAIN)

CV = CountVectorizer(binary=True)
CV.fit(CHAT_TRAIN)
X = CV.transform(CHAT_TRAIN)
X_TEST = CV.transform(CHAT_TEST)

# Using Logistic Regression model to train

TARGET = [1 if i < 21 else 0 for i in range(45)]

X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X, TARGET, train_size=0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_TRAIN, Y_TRAIN)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(Y_VAL, lr.predict(X_VAL))))

FINAL_MODEL = LogisticRegression(C=0.05)
FINAL_MODEL.fit(X, TARGET)
print("Final Accuracy: %s" % accuracy_score(TARGET, FINAL_MODEL.predict(X_TEST)))

FEATURE_TO_COEF = {
    word: coef for word, coef in zip(
        CV.get_feature_names(), FINAL_MODEL.coef_[0]
    )
}
for best_positive in sorted(FEATURE_TO_COEF.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(best_positive)

for best_negative in sorted(FEATURE_TO_COEF.items(), key=lambda x: x[1])[:5]:
    print(best_negative)

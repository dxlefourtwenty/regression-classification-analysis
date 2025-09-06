import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# get the same results each time
np.random.seed(0)

# load training data
data = pd.read_csv("./data.csv")
comments = data["comment_text"]
target = (data["target"] > 0.7).astype(int)

# break into training and test sets
comments_train, comments_test, y_train, y_test = train_test_split(
    comments, target, test_size=0.30, stratify=target)

# get vocabulary from training data
vectorizer = CountVectorizer()
vectorizer.fit(comments_train)

# get word counts for training and test sets
X_train = vectorizer.transform(comments_train)
X_test = vectorizer.transform(comments_test)

# preview dataset
if False:
    print("\nPreviewing dataset...\n")
    print("Sample toxic comment:", comments_train.iloc[22])
    print("Sample non-toxic comment:", comments_train.iloc[17])


# train the LR model and evaluate performance on data provided
classifier = LogisticRegression(max_iter=2000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

# function classifies any string using Logistic Regression
def classifyLogReg(string, investigate=False):
    prediction = classifier.predict(vectorizer.transform([string]))[0]
    if prediction == 0:
        print("NOT TOXIC:", string)
    else:
        print("TOXIC:", string)

# test LR model with example strings (comments)
example_toxic = "This post is dumb"
example_toxic_2 = "Your outfit sucks bruh"
example_non_toxic = "I love your hair"
example_non_toxic_2 = "Your car is cool dude"
if False:
    print("\n(LR Model) Testing with typical comments on social media posts:\n")
    classifyLogReg(example_toxic)
    classifyLogReg(example_non_toxic)
    classifyLogReg(example_non_toxic_2)
    classifyLogReg(example_toxic_2)


# understanding the models with coefficients
if False:
    print("\nTop coefficients within the dataset:\n")
    coefficients = pd.DataFrame({"word" : sorted(list(vectorizer.
        vocabulary_.keys())), "coeff": classifier.coef_[0]})
    coefficients.sort_values(by=['coeff']).tail(10)
    # print highest coeffs in pairs
    top_pairs = (
        coefficients.sort_values(by='coeff')
        .tail(10)
        [['word', 'coeff']]
        .values.tolist()
    )
    print(top_pairs)


# the bias of the LR model
bias_one = "I have a muslim friend"
bias_two = "I have a christian friend"
bias_three = "I have a white friend"
bias_four = "I have a black friend"
if False:
    print("\n(LR Model) Testing the model's potential bias:\n")
    classifyLogReg(bias_one)
    classifyLogReg(bias_two)
    classifyLogReg(bias_three)
    classifyLogReg(bias_four)


# train model #2 Linear SVC
svm = LinearSVC()
svm.fit(X_train, y_train)

# function classifies any string with SVC
def classifyLinSVC(string, investigate=False):
    prediction = svm.predict(vectorizer.transform([string]))[0]
    if prediction == 0:
        print("NOT TOXIC:", string)
    else:
        print("TOXIC:", string)

# test LSVC Model with example strings
if True:
    print("\n(LSVC Model) Testing with typical comments on social media posts:\n")
    classifyLinSVC(example_non_toxic)
    classifyLinSVC(example_toxic)
    classifyLinSVC(example_non_toxic_2)
    classifyLinSVC(example_toxic_2)

# test LSVC Model bias
if True:
    print("\n(LSVC Model) Testing the model's potential bias:\n")
    classifyLinSVC(bias_one)
    classifyLinSVC(bias_two)
    classifyLinSVC(bias_three)
    classifyLinSVC(bias_four)



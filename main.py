import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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
    print("Sample toxic comment:", comments_train.iloc[22])
    print("Sample non-toxic comment:", comments_train.iloc[17])


# train the model and evaluate performance on data provided
classifier = LogisticRegression(max_iter=2000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

# function classifies any string
def classify(string, investigate=False):
    prediction = classifier.predict(vectorizer.transform([string]))[0]
    if prediction == 0:
        print("NOT TOXIC:", string)
    else:
        print("TOXIC:", string)

# test model with strings (comments)
if False:
    example_toxic = "This post is dumb"
    example_toxic_2 = "Your outfit sucks bruh"
    example_non_toxic = "I love your hair"
    example_non_toxic_2 = "Your car is cool dude"
    classify(example_toxic)
    classify(example_non_toxic)
    classify(example_non_toxic_2)
    classify(example_toxic_2)


# understanding the model with coefficients
if False:
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


# the bias of the model
if True:
    comment_one = "I have a muslim friend"
    comment_two = "I have a christian friend"
    comment_three = "I have a white friend"
    comment_four = "I have a black friend"
    classify(comment_one)
    classify(comment_two)
    classify(comment_three)
    classify(comment_four)


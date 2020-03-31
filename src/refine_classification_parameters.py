import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from time import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


import nltk
from nltk.corpus import stopwords

reviews_with_beer_info = pd.read_pickle('../data/beers_reviews_with_sentiment.pkl')

X_large, X_small, y_large, y_small = train_test_split(reviews_with_beer_info.text, 
                                    reviews_with_beer_info.sentiment, 
                                    test_size=0.5, 
                                    random_state=284)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_small, 
                                                    y_small, 
                                                    test_size=0.3, 
                                                    random_state=284)


extra_words = ['beer', 'oz', 'ml', 'write', 'review']
all_stop_words = stopwords.words('english') + extra_words
print(all_stop_words)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=all_stop_words)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)



def fit_predict(clf, name):
    # train
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    print(accuracy_score(y_test, pred))
    precision, recall, f1score, support = precision_recall_fscore_support(y_test, pred, pos_label='negative', average='binary')
    return name, precision, recall, f1score, support, train_time, test_time

def sentiment_from_proba(x):
    if x >= 0.5:
        return 'positive'
    else:
        return 'negative'


def fit_multiple(clfs, name, weights = []):
    if len(weights) == 0:
        weights = np.full(len(clfs), 0.5)
    preds_proba = []
    train_time = 0
    test_time = 0
    for clf in clfs:
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time += time() - t0
        print("train time: %0.3fs" % train_time)
        
        t0 = time()

        preds_proba.append(clf.predict_proba(X_test)[:, 1])
        test_time += time() - t0

    probas = np.average(preds_proba, axis=0, weights = weights)
    pred = [sentiment_from_proba(x) for x in probas]

    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    print(accuracy_score(y_test, pred))
    precision, recall, f1score, support = precision_recall_fscore_support(y_test, pred, pos_label='negative', average='binary')
    return name, precision, recall, f1score, support, train_time, test_time





##############################################################################
# Add plots
def make_plot(results, img_name):

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(5)]

    clf_names, precision, recall, f1score, support = results
    # training_time = np.array(training_time) / np.max(training_time)
    # test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, (len(indices)*.6) +2))
    plt.title("Score")
    plt.barh(indices, f1score, .2, label="f1 score", color='navy')
    plt.barh(indices + .3, precision, .2, label="precision",
             color='c')
    plt.barh(indices + .6, recall, .2, label="recall", color='darkorange')
    plt.yticks(())
    plt.xlim(0,1)
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.savefig(img_name)


from sklearn.naive_bayes import MultinomialNB, ComplementNB

nb_results = []

for alpha in [1.0, 0.1, 1e-2, 1e-3, 1e-4]:
    nb_results.append(fit_predict(MultinomialNB(alpha=alpha), 'MultinomialNB alpha=%f' % alpha))

for alpha in [1.0, 0.1, 1e-2, 1e-3, 1e-4]:
    nb_results.append(fit_predict(ComplementNB(alpha=alpha), 'ComplementNB alpha=%f' % alpha))

make_plot(nb_results, '../images/nb_results.png')

from sklearn.svm import LinearSVC

svm_results = []
for penalty in ['l1', 'l2']:
    for c in [1.0, 0.1, 0.01]:
        svm_results.append(fit_predict(LinearSVC(penalty=penalty, dual=False, tol=1e-4, C=c), 'LinearSVC %s, c=%f' % (penalty, c)))

make_plot(svm_results, '../images/svm_results.png')

results = []
from sklearn.linear_model import PassiveAggressiveClassifier
for c in [1.0, 0.1, 0.01]:
    results.append(fit_predict(PassiveAggressiveClassifier(C=c), 'PassiveAggressive c=%f' % c))
make_plot(results, '../images/passiveaggresive_results.png')

combo_results = []

combo_results.append(fit_multiple([MultinomialNB(alpha=0.1), ComplementNB(alpha=0.1)],
                            'MultinomialNB+ComplementNB'))

combo_results.append(fit_multiple([MultinomialNB(alpha=0.1), ComplementNB(alpha=0.1)],
                            '.25MultinomialNB+.75ComplementNB', weights = [0.25, 0.75]))

combo_results.append(fit_multiple([MultinomialNB(alpha=0.1), ComplementNB(alpha=0.1)],
                            '.75MultinomialNB+.25ComplementNB', weights = [0.75, 0.25]))

make_plot(combo_results, '../images/combo_results.png')
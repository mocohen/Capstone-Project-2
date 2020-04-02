# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def getSentiment(x):
    if x < 3.0:
        return 'negative'
    else:
        return 'positive'


def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    s += "%(n_train)6d train reviews (%(n_train_pos)6d positive) " % stats
    s += "in %.2fs (%5d reviews/s)" % (duration, stats['n_train'] / duration)
    return s


def sentiment_from_proba(x):
    if x >= 0.5:
        return 'positive'
    else:
        return 'negative'


def train_model_in_chunks(file_iter, vectorizer, partial_fit_classifiers, test_size=0.3, report_print_frequency=100, random_state=42):

    # set up variables for running
    total_other_time= 0.0
    cls_stats = {}
    total_vect_time = 0.0

    x_test_data_chunks = []
    y_test_data_chunks = []

    cls_stats = {}

    report_print_frequency = 100

    for cls_name in partial_fit_classifiers:
        stats = {'n_train': 0, 'n_train_pos': 0,
                 't0': time.time(),
                 'total_fit_time': 0.0}
        cls_stats[cls_name] = stats


    for i, chunk in enumerate(file_iter):
        #get start time
        tick = time.time()

        # drop null values
        chunk.dropna(inplace=True)

        #add sentiment to dataframe
        chunk['sentiment'] = chunk.overall.apply(getSentiment)
        
        # check to make sure there are values in the dataframe
        if len(chunk) > 0:

            #split data into test and training sets
            X_train_text, X_test_text, y_train, y_test = train_test_split(chunk.text, 
                                                chunk.sentiment, 
                                                test_size=test_size, 
                                                random_state=random_state)



            # vectorize data
            tick = time.time()

            X_train = vectorizer.transform(X_train_text)
            X_test = vectorizer.transform(X_test_text)
            total_vect_time += time.time() - tick

            x_test_data_chunks.append(X_test)
            y_test_data_chunks.append(y_test)

            # loop through classifiers and fit
            for cls_name, cls in partial_fit_classifiers.items():
                    tick = time.time()
                    # update estimator with examples in the current mini-batch
                    cls.partial_fit(X_train, y_train, classes=['positive', 'negative'])

                    # accumulate test accuracy stats
                    cls_stats[cls_name]['total_fit_time'] += time.time() - tick
                    cls_stats[cls_name]['n_train'] += X_train.shape[0]
                    cls_stats[cls_name]['n_train_pos'] += sum(y_train == 'positive')


                    if i % report_print_frequency == 0:
                        print(progress(cls_name, cls_stats[cls_name]))

        if i % report_print_frequency == 0:
            print('\n')   

    return partial_fit_classifiers, vectorizer, cls_stats, (x_test_data_chunks, y_test_data_chunks) 


def predict_on_chunks(test_chunks, cls):
    preds = []

    for i, x_test in enumerate(test_chunks):
        preds.append(cls.predict(x_test))

    return np.concatenate(preds, axis=1)


def predict_proba_on_chunks(test_chunks, cls):
    preds = []

    for i, x_test in enumerate(test_chunks):
        preds.append(cls.predict_proba(x_test)[:, 1])

    return np.concatenate(preds)


def blend_models(probas_list, weights):
    return np.average(probas_list, axis=0, weights=(.1, .9))


def print_results(cls_name, predictions, true):
    print(cls_name)
    precision, recall, f1score, support = precision_recall_fscore_support(true, 
                                                                          predictions, 
                                                                          pos_label='negative', 
                                                                          average='binary')    
    
    print(confusion_matrix(true,predictions))
    print(classification_report(true,predictions))
    print(accuracy_score(true, predictions))
    print('precision:', precision)
    print('recall:', recall)
    print('f1 score:', f1score)


def print_proba_results(cls_name, probas, true):
    print_results(cls_name, [sentiment_from_proba(x) for x in probas], true)


import sys

import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from TSA.Preproc.Preproc import Preproc
from datetime import datetime
from time import time

pp = Preproc()

pp.loadCsv("TSA/datasets/SemEval/4A-English/", "preproc_SemEval.csv")

df = pp.get_twitter_df()

X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.lable, test_size=0.2, random_state=0)

target_names = ['Positive', 'Negative']

# SemEval vocab size 10272

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('kbest', SelectKBest(chi2)),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': (None, 'l1', 'l2'),
    'kbest__k': (3000, 6000, 8000, 'all'),
    'clf__alpha': (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
    'clf__max_iter': (500, 1000, 1500),
    'clf__penalty': ('l2', 'l1', 'elasticnet'),
}

if __name__ == "__main__":

    log_name = "best_params_svm_se.log"
    old_stdout = sys.stdout

    if os.path.isfile("best_params_svm_se.log"):
        file_permission = "w"
    else:
        file_permission = "a"

    log_file = open("../../../SVM/best_params_svm_se.log", file_permission)
    sys.stdout = log_file  # redirect output to logfile

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("\nPerforming grid search at " + str(datetime.utcnow()))
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)

    t0 = time()
    grid_search.fit(X_train,y_train)

    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # clean up use of log file
    sys.stdout = old_stdout
    log_file.close()
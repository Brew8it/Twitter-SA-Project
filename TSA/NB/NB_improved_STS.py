import datetime
import os
import sys

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

from TSA.Preproc.Preproc import Preproc


def train_NB():
    pp = Preproc()

    pp.loadCsv("TSA/datasets/STS/", "preproc_STS.csv")

    print("Data is loaded :: " + str(datetime.datetime.utcnow()))

    df = pp.get_twitter_df()

    print("Data is cleand time for splitting :: " + str(datetime.datetime.utcnow()))

    X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.lable, test_size=0.2, random_state=0)

    target_names = ['Positive', 'Negative']

    log_name = "NB_imp_STS.log"
    old_stdout = sys.stdout

    if os.path.isfile("NB_imp_STS.log"):
        file_permission = "w"
    else:
        file_permission = "a"

    log_file = open("../../NB/NB_imp_STS.log", file_permission)
    sys.stdout = log_file  # redirect output to logfile

    # Train the model
    nb_improved_clf = Pipeline([
        ('vect', CountVectorizer(max_df=0.5, ngram_range=(1, 3))),
        ('kbest', SelectKBest(chi2, k='all')),
        ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
        ('clf', MultinomialNB()),
    ])
    print("\nTrain the model :: " + str(datetime.datetime.utcnow()))
    nb_improved_clf.fit(X_train, y_train)

    print("Training finished :: " + str(datetime.datetime.utcnow()) + "\n")

    # Test the model
    predicted = nb_improved_clf.predict(X_test)

    print("\nPerforming NB_STS testing at " + str(datetime.datetime.utcnow()))

    # Print evaluation metrics
    print(metrics.classification_report(y_test, predicted, target_names=target_names, digits=3))

    sys.stdout = old_stdout
    log_file.close()

    joblib.dump(nb_improved_clf, "../../TrainedModels/NB_imp_STS.pkl")

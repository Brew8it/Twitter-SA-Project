from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

import datetime

from TSA.Preproc.Preproc import Preproc


def train_SVM():
    pp = Preproc()

    # pp.loadCsv("TSA/datasets/SemEval/4A-English/", "SemEval.csv")
    pp.loadCsv("TSA/datasets/STS/", "STS.csv")

    print("Data is loaded :: " + str(datetime.datetime.utcnow()))

    pp.clean_data()
    df = pp.get_twitter_df()

    print("Data is cleand time for splitting :: " + str(datetime.datetime.utcnow()))

    X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.lable, test_size=0.2, random_state=0)

    target_names = ['Positive', 'Negative']

    print("Train the model :: " + str(datetime.datetime.utcnow()))

    # Train the model
    svm_unigram_clf = Pipeline([('vect', CountVectorizer()),
                                ('clf', SGDClassifier())])
    svm_unigram_clf.fit(X_train, y_train)

    print("Testing the model :: " + str(datetime.datetime.utcnow()))

    # Test the model
    predicted = svm_unigram_clf.predict(X_test)

    # Print evaluation metrics
    print(metrics.classification_report(y_test, predicted, target_names=target_names))

    # joblib.dump(svm_unigram_clf, "../../../TrainedModels/SVM_base_SemEval.pkl")

    joblib.dump(svm_unigram_clf, "../../TrainedModels/SVM_base_STS.pkl")

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

import datetime


from preproc import preproc

pp = preproc.preProc()

# pp.loadCsv("../datasets/SemEval/4A-English/", "SemEval.csv")
pp.loadCsv("../datasets/STS/", "STS.csv")

print("Data is loaded :: " + str(datetime.datetime.utcnow()))

pp.clean_data()
df = pp.get_twitter_df()

print("Data is cleand time for splitting :: " + str(datetime.datetime.utcnow()))


X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.lable, test_size=0.2, random_state=0)


target_names = ['Positive', 'Negative']

print("Train the model :: " + str(datetime.datetime.utcnow()))

# Train the model
svm_unigram_clf = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', SGDClassifier())])
svm_unigram_clf.fit(X_train, y_train)

print("Testing the model :: " + str(datetime.datetime.utcnow()))

# Test the model
predicted = svm_unigram_clf.predict(X_test)

# Print evaluation metrics
print(metrics.classification_report(y_test, predicted, target_names=target_names))

#joblib.dump(svm_unigram_clf, "../../../models/SVM_base.pkl")

joblib.dump(svm_unigram_clf, "../../models/SVM_base_STS.pkl")




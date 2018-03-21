import datetime

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV

from TSA.Preproc.Preproc import Preproc

from time import time




pp = Preproc()
pp.loadCsv("../datasets/STS/", "preproc_STS.csv")

print("Data is loaded :: " + str(datetime.datetime.utcnow()))

#pp.clean_data()
df = pp.get_twitter_df()

print("Data is cleand time for splitting :: " + str(datetime.datetime.utcnow()))

X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.lable, test_size=0.2, random_state=0)

# print(X_train.shape)

target_names = ['Positive', 'Negative']


#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(X_train)
#print(X_train_counts.shape)
#print(len(count_vect.vocabulary_.keys()))

# STS vocab size 200000
# SemEval vocab size 10272


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('kbest', SelectKBest(chi2)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': (None, 'l1', 'l2'),
    'kbest__k': (30000, 50000, 100000, 130000, 'all'),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
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

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics

target_names = ['Positive', 'Negative']

# Train the model
nb_unigram_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
nb_unigram_clf.fit()

# Test the model
predicted = nb_unigram_clf.predict(test_data)

# Print evaluation metrics
print(metrics.classification_report(y_true, predicted, target_names=target_names))



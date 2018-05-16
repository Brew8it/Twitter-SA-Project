# Twitter Sentiment Analysis
#### Comparison of classification models trained on different data sets.

The purpose of this study is to evaluate three classification algorithms in ma-
chine learning and how the labeling of a data set affects classification models per-
formance for Twitter sentiment analysis.
Naive Bayes, Support Vector Machine
and Convolutional Neural Network are the classification algorithms that have been
evaluated. For each classification algorithm, two classification models have been
trained and tested on two separate data sets: Stanford Twitter Sentiment and Se-
mEval. What separates the two data sets, in addition to the content of the twitter
posts, is the labeling method and the amount of twitter posts.
The evaluation
has been done according to the performance of the classification models on the
respective data sets, training time and how complicated they were to implement.
The results show that all models trained and tested on SemEval achieved a
higher performance than those trained and tested on Stanford Twitter Sentiment.
The Convolutional Neural Network models achieved the best results over both data
sets. However, a Convolutional Neural Network is more complicated to implement
and the training time is significantly longer than Naive Bayes and Support Vector
Machine.

## Installation

From project folder run the following command:
``` 
pip install -r requirements.txt
```
After installing the python packages you need to install following subpackages for NLTK:
```
python3
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('sentiwordnet')
>>> nltk.download('wordnet')
>>> nltk.download('averaged_perceptron_tagger')
```
## Run 

To run the project, run the run.py script. And the following options will be presented:

* Train Naive Bayes
* Train SVM
* Train CNN
* Start webserver for GUI
* Test Pred
* Save preproc to csv
* Train improved NB_SE
* Train improved NB_STS
* Train improved SVM_SE
* Train improved SVM_STS
* Run Lexicon


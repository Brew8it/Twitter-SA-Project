import numpy
from sklearn.externals import joblib
import pandas as pd

from miner import miner
from preproc import preproc
pd.options.display.max_colwidth = 250


class Prediction:
    #modelnames = ["../models/SVM_base.pkl", "../models/NB_base.pkl"]
    modelnames = ["../../models/SVM_base.pkl", "../../models/NB_base.pkl"]
    def __init__(self):
        self.clflist = []
        self.load_models()
        self.twitterMiner = miner.TwitterMiner()
        self.twitterDF = pd.DataFrame
        self.cleanTwitterDF = pd.DataFrame
        self.preProcess = preproc.preProc()
        self.numberOfposNeg = []

    def load_models(self):
        for clf in self.modelnames:
            self.clflist.append(joblib.load(clf))

    def make_predictions(self, name, numTweets):
        self.get_twitter_data(name, numTweets)
        self.make_preproc()
        self.predict(self.cleanTwitterDF.tweet)
        return self.numberOfposNeg

    def make_posneg_list(self, predlist):
        #pos, #neg # nodata
        _, counts = numpy.unique(predlist, return_counts=True)
        lst = list(reversed(counts))
        lst.append(0)
        self.numberOfposNeg.append(lst)


    def predict(self, text):
        for clf in self.clflist:
            pred = clf.predict(text)
            self.make_posneg_list(pred)
            #print(self.twitterDF.tweet)
            # spara till DB istället?
        print(self.numberOfposNeg)

    def get_twitter_data(self, name, numTweets):
        self.twitterDF = self.twitterMiner.collect_tweets_from_user_feed(name, numTweets)

    def make_preproc(self):
        self.preProcess.loadOwnDataFrame(self.twitterDF)
        self.preProcess.clean_data()
        self.cleanTwitterDF = self.preProcess.get_twitter_df()


if __name__ == "__main__":
    prediction = Prediction()
    prediction.make_predictions("danlevene", "10")





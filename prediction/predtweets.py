import numpy
from sklearn.externals import joblib
import pandas as pd

from miner import miner
from preproc import preproc
pd.options.display.max_colwidth = 250


class Prediction:
    #modelnames = ["../models/NB_base.pkl", "../models/NB_base_STS.pkl", "../models/SVM_base.pkl", "../models/SVM_base_STS.pkl"]
    modelnames = ["../../models/NB_base.pkl", "../../models/NB_base_STS.pkl", "../../models/SVM_base.pkl", "../../models/SVM_base_STS.pkl"]
    def __init__(self):
        self.clflist = []
        self.load_models()
        self.twitterMiner = miner.TwitterMiner()
        self.twitterDF = pd.DataFrame
        self.cleanTwitterDF = pd.DataFrame
        self.preProcess = preproc.preProc()
        self.numberOfposNeg = []
        self.pred_list = []
        self.db_list = []

    def load_models(self):
        for clf in self.modelnames:
            self.clflist.append(joblib.load(clf))

    def db_make_predictions(self, name, numTweets):
        self.get_twitter_data(name,numTweets)
        self.make_preproc()
        self.predict(self.cleanTwitterDF.tweet)
        self.make_db_list()
        return self.db_list

    # for internal bib testing...
    def make_predictions(self, name, numTweets):
        self.get_twitter_data(name, numTweets)
        self.make_preproc()
        self.predict(self.cleanTwitterDF.tweet)
        return self.numberOfposNeg

    def make_db_list(self):
        # change this dependeing on how many models!!
        # NBSE, NBSTS, SVMSE, SVMSTS
        NBSE = self.pred_list[0]
        NBSTS = self.pred_list[1]
        SVMSE = self.pred_list[2]
        SVMSTS = self.pred_list[3]
        self.db_list = [list(e) for e in zip(list(self.twitterDF.tweet),NBSE, NBSTS, SVMSE, SVMSTS)]

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
            # Convert to int list so it will fit into DB.
            self.pred_list.append([int(i) for i in pred])

    def get_twitter_data(self, name, numTweets):
        self.twitterDF = self.twitterMiner.collect_tweets_from_user_feed(name, numTweets)

    def make_preproc(self):
        self.preProcess.loadOwnDataFrame(self.twitterDF)
        self.preProcess.clean_data()
        self.cleanTwitterDF = self.preProcess.get_twitter_df()


if __name__ == "__main__":
    prediction = Prediction()
    prediction.db_make_predictions("danlevene", "10")





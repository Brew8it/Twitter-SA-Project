from sklearn.externals import joblib

from miner import miner
from preproc import preproc


class Prediction:
    modelnames = ["../models/SVM_base.pkl", "../models/NB_base.pkl"]

    def __init__(self):
        self.clflist = []
        self.load_models()

    def load_models(self):
        for clf in self.modelnames:
            self.clflist.append(joblib.load(clf))

    def make_predictions(self, text):
        for clf in self.clflist:
            pred = clf.predict(text)
            print(pred)


if __name__ == "__main__":
    twitterMiner = miner.TwitterMiner()
    preProcess = preproc.preProc()
    prediction = Prediction()

    name = input("Enter a name: ")
    numTweets = input("Enter how many Tweets to download: ")
    print(name + "::." + numTweets)
    twitterDf = twitterMiner.collect_tweets_from_user_feed(name, numTweets)
    print(twitterDf.tweet)
    preProcess.loadOwnDataFrame(twitterDf)
    preProcess.clean_data()
    twitterDf = preProcess.get_twitter_df()

    prediction.make_predictions(twitterDf.tweet)





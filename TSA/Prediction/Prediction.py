import numpy
from sklearn.externals import joblib
import pandas as pd
import numpy as np

from TSA.TwitterMiner import TwitterMiner
from TSA.Preproc import Preproc

pd.options.display.max_colwidth = 250

from keras.models import model_from_json

from TSA.CNN import data_handler


class Prediction:
    modelnames = ["TSA/TrainedModels/NB_imp_SemEval.pkl", "TSA/TrainedModels/NB_imp_STS.pkl",
                  "TSA/TrainedModels/SVM_imp_SemEval.pkl",
                  "TSA/TrainedModels/SVM_imp_STS.pkl"]

    cnnmodels = [["TSA/TrainedModels/CNN_base_SemEval.json", "TSA/TrainedModels/CNN_base_SemEval_w.h5"], ["TSA/TrainedModels/CNN_base_STS.json", "TSA/TrainedModels/CNN_base_STS_w.h5"]]

    def __init__(self):
        self.clflist = []
        self.cnnlist = []
        self.load_models()
        self.load_cnn_models()
        self.twitterMiner = TwitterMiner.TwitterMiner()
        self.twitterDF = pd.DataFrame
        self.cleanTwitterDF = pd.DataFrame
        self.preProcess = Preproc.Preproc()
        self.numberOfposNeg = []
        self.pred_list = []
        self.db_list = []
        self.clean_DF_cnn = []

    def load_models(self):
        for clf in self.modelnames:
            self.clflist.append(joblib.load(clf))

    def load_cnn_models(self):
        for clf in self.cnnmodels:
            json_file = open(clf[0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            # and create a model from that
            model = model_from_json(loaded_model_json)
            # and weight your nodes with your saved values
            model.load_weights(clf[1])
            self.cnnlist.append(model)

    def clear_lists(self):
        self.numberOfposNeg = []
        self.pred_list = []
        self.db_list = []
        self.clean_DF_cnn = []

    def db_make_predictions(self, name, numTweets):
        self.get_twitter_data(name, numTweets)
        self.make_preproc()
        self.nb_svm_predict(self.cleanTwitterDF.tweet)
        # make cnn preproc
        self.make_cnn_preproc()
        # make cnn_pred
        self.cnn_predict(self.clean_DF_cnn)
        self.make_average_prediction()
        self.make_db_list()
        return self.db_list

    # for internal bib testing...
    def make_predictions(self, name, numTweets):
        self.get_twitter_data(name, numTweets)
        self.make_preproc()
        self.nb_svm_predict(self.cleanTwitterDF.tweet)
        return self.numberOfposNeg

    def make_average_prediction(self):
        sum = []
        for i in range(len(self.pred_list[0])):
            _sum = 0
            for j in range(len(self.pred_list)):
                _sum += self.pred_list[j][i]
            avg = _sum/len(self.pred_list)
            sum.append(float("{0:.2f}".format(avg)))
        self.pred_list.append(sum)



    def make_db_list(self):
        # change this dependeing on how many models!!
        # NBSE, NBSTS, SVMSE, SVMSTS
        NBSE = self.pred_list[0]
        NBSTS = self.pred_list[1]
        SVMSE = self.pred_list[2]
        SVMSTS = self.pred_list[3]
        CNNSE = self.pred_list[4]
        CNNSTS = self.pred_list[5]
        AVG = self.pred_list[6]
        ## until models are done

        # CNNSE = []
        # CNNSTS = []
        #
        # for i in range(len(self.pred_list[0])):
        #     CNNSTS.append(0)
        #   #  CNNSE.append(0)
        self.db_list = [list(e) for e in zip(list(self.twitterDF.tweet), NBSE, NBSTS, SVMSE, SVMSTS, CNNSE, CNNSTS, AVG)]

    def make_posneg_list(self, predlist):
        # pos, #neg # nodata
        _, counts = numpy.unique(predlist, return_counts=True)
        lst = list(reversed(counts))
        lst.append(0)
        self.numberOfposNeg.append(lst)

    def nb_svm_predict(self, text):
        for clf in self.clflist:
            pred = clf.predict(text)
            self.make_posneg_list(pred)
            # Convert to int list so it will fit into DB.
            self.pred_list.append([int(i) for i in pred])

    def cnn_make_predict_lable(self, pred):
        labels = [0, 1]
        arglist = []
        # Swap from [0.423, 0.677] -> lable = 0
        for p in pred:
            arglist.append(labels[np.argmax(p)])

        return arglist




    def cnn_predict(self, text):
        for clf in self.cnnlist:
            pred = clf.predict(text)
            pred_remade = self.cnn_make_predict_lable(pred)
            self.make_posneg_list(pred_remade)
            # Convert to int list so it will fit into DB.
            self.pred_list.append([int(i) for i in pred_remade])

    def get_twitter_data(self, name, numTweets):
        self.twitterDF = self.twitterMiner.collect_tweets_from_user_feed(name, numTweets)

    def make_preproc(self):
        self.preProcess.loadOwnDataFrame(self.twitterDF)
        self.preProcess.clean_data()
        self.cleanTwitterDF = self.preProcess.get_twitter_df()

    def make_cnn_preproc(self):
        self.clean_DF_cnn = data_handler.pred_load_data(self.preProcess.get_twitter_df())


def main():
    # add support for input username and number of tweets.
    prediction = Prediction()
    data = prediction.db_make_predictions("realdonaldtrump", "2")
    print(data)

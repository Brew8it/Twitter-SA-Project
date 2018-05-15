from sklearn.externals import joblib


class NaiveBayes:
    def __init__(self):
        self.model = joblib.load("TSA/TrainedModels/NB_imp_SemEval.pkl")

    def predict(self, text):
        try:
            return self.model.predict(text)
        except:
            print("Data is not foromated correctly. Try preprocessing it...")
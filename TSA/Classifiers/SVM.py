from sklearn.externals import joblib


class SupportVectorMachine:
    def __init__(self):
        self.model = joblib.load("TSA/TrainedModels/SVM_imp_SemEval.pkl")

    def predict(self, text):
        try:
            return self.model.predict(text)
        except:
            print("Data is not foromated correctly. Try preprocessing it...")
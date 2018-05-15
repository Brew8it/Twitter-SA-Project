from keras.models import model_from_json


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.model = self.load_cnn()

    def load_cnn(self):
        json_file = open("TSA/TrainedModels/CNN_base_SemEval.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # and create a model from that
        model = model_from_json(loaded_model_json)
        # and weight your nodes with your saved values
        model.load_weights("TSA/TrainedModels/CNN_base_SemEval_w.h5")
        return  model

    def predict(self, text):
        try:
            return self.model.predict(text)
        except:
            print("Data is not foromated correctly. Try preprocessing it...")
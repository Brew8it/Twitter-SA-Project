# import the Flask class from the flask module
from flask import Flask, render_template, request, url_for, redirect
# from prediction import predtweets
from TSA.Prediction import Prediction

from TSA.GUI.database.db import *

# create the application object
app = Flask(__name__)


# use decorators to link the function to a url
@app.route('/', methods=["GET", "POST"])
def home():
    tdata = ["test", "test2"]

    if request.method == "POST":
        username = request.form['username']
        numberOfTweets = request.form['numberOfTweets']

        clear_searches()  # Clear recent search
        prediction = Prediction.Prediction()
        predicted_tweets = prediction.db_make_predictions(username, numberOfTweets)
        insert_search_with_tweets(username, numberOfTweets, predicted_tweets)

        return redirect(url_for('dashboard'))
    else:

        return render_template('index.html')


@app.route('/dashboard')
# Pass parameter to know if we should pass 0,0,100 / passed search or do new search
def dashboard():
    sentiment_result = get_posneg()

    series = {
        'NBSE': {'series': sentiment_result["NBSE"]},
        'NBSTS': {'series': sentiment_result["NBSTS"]},
        'SVMSE': {'series': sentiment_result["SVMSE"]},
        'SVMSTS': {'series': sentiment_result["SVMSTS"]},
        'CNNSE': {'series': sentiment_result["CNNSE"]},
        'CNNSTS': {'series': sentiment_result["CNNSTS"]},
    }

    #data = [['Perhaps at no time in history have the business fundamentals of U.S. companies been better than they are today!Perhaps at no time in history have the business fundamentals of U.S. companies been better than they are today!', 1, 1, 1, 1], ['Perhaps at no time in history have the business fundamentals of U.S. companies been better than they are today!', 1, 1, 1, 1]]
    data = get_tweets_records()

    return render_template('dashboard.html', **series, tdata=data)


# start the server with the 'run()' method
def main():
    # for debug = debug=True
    app.run()

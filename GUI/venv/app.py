# import the Flask class from the flask module
from flask import Flask, render_template, request, url_for, redirect
from prediction import predtweets
from db import *

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
        prediction = predtweets.Prediction()
        predicted_tweets = prediction.db_make_predictions(username, numberOfTweets)
        insert_search_with_tweets(username, numberOfTweets, predicted_tweets)

        return redirect(url_for('dashboard'))
    else:

        return render_template('index.html')


@app.route('/dashboard')
# Pass parameter to know if we should pass 0,0,100 / passed search or do new search
def dashboard():

    sentiment_result = get_posneg()

    print(sentiment_result)

    series = {
        'NBSE': {'series': sentiment_result["NBSE"]},
        'NBSTS': {'series': sentiment_result["NBSTS"]},
        'SVMSE': {'series': sentiment_result["SVMSE"]},
        'SVMSTS': {'series': sentiment_result["SVMSTS"]}
    }

    data = "tssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttsst"

    return render_template('dashboard.html', **series, item=data)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)

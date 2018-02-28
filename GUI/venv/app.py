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

        prediction = predtweets.Prediction()

        predicted_tweets = prediction.db_make_predictions(username, numberOfTweets)

        print(predicted_tweets)

        insert_search_with_tweets(username, numberOfTweets, predicted_tweets)

        print("Search")
        print(get_search_records())
        print("Tweets")
        print(get_tweets_records())

        return redirect(url_for('dashboard', username=username, numberOfTweets=numberOfTweets))
    else:

        return render_template('index.html')


@app.route('/dashboard')
# Pass parameter to know if we should pass 0,0,100 / passed search or do new search
def dashboard():
    #fake data
    #series = [20, 40, 0]
    #nodata

    username = request.args['username']
    numberOfTweets = request.args['numberOfTweets']

    print(username)

    prediction = predtweets.Prediction()

    lst = prediction.make_predictions(username, numberOfTweets)

    print(lst)

    series = {
        'NBSE': {'series': lst[0]},
        'NBSTS': {'series': [0, 0, 1]},
        'SVMSE': {'series': lst[1]},
        'SVMSTS': {'series': [0, 0, 1]}
    }
    data = "tssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttsst"

    return render_template('dashboard.html', **series, item=data)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)

# import the Flask class from the flask module
from flask import Flask, render_template, request, url_for, redirect

# create the application object
app = Flask(__name__)


# use decorators to link the function to a url
@app.route('/', methods=["GET", "POST"])
def home():
    tdata = ["test", "test2"]

    if request.method == "POST":
        username = request.form['username']
        numberOfTweets = request.form['numberOfTweets']

        return redirect(url_for('dashboard'))
    else:

        return render_template('index.html')


@app.route('/dashboard')
# Pass parameter to know if we should pass 0,0,100 / passed search or do new search
def dashboard():
    #fake data
    series = [20, 40, 0]
    #nodata
    series = [0, 0, 100]
    data = "tssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttssttsst"

    return render_template('dashboard.html', series=series, item=data)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)

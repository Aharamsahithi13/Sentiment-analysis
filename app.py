from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

def predict_sentiment(text):
    analysis = TextBlob(text)
    # Use TextBlob's polarity to predict sentiment
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['textInput']
        sentiment = predict_sentiment(text)
        return render_template('index.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == "_main_":
    app.run(debug=True)
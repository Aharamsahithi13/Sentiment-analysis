#sentimentanalyzer.py
from textblob import TextBlob

class SentimentAnalysis:

    def __init__(self):
        # Initialize TextBlob sentiment analyzer
        self.analyzer = TextBlob()

    def predict_sentiment(self, text):
        # Analyze sentiment using TextBlob
        sentiment = self.analyzer(text).sentiment

        # Determine sentiment label based on polarity
        if sentiment.polarity > 0:
            sentiment_label = "positive"
        elif sentiment.polarity == 0:
            sentiment_label = "neutral"
        else:
            sentiment_label = "negative"

        # Return sentiment label
        return sentiment_label

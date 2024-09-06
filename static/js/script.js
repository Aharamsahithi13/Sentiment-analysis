function analyzeSentiment() {
    var textInput = document.getElementById('textInput').value;

    // Perform simple sentiment analysis using AFINN-111 wordlist
    var words = textInput.split(/\s+/);
    var sentimentScore = 0;

    for (var i = 0; i < words.length; i++) {
        var word = words[i].toLowerCase().replace(/[^\w\s]/gi, ''); // Remove punctuation
        if (AFINN.hasOwnProperty(word)) {
            sentimentScore += parseInt(AFINN[word]);
        }
    }

    // Display result
    var resultDiv = document.getElementById('result');
    if (sentimentScore > 0) {
        resultDiv.innerHTML = 'Sentiment: Positive';
        resultDiv.style.color = '#4caf50'; // Green color for positive sentiment
    } else if (sentimentScore < 0) {
        resultDiv.innerHTML = 'Sentiment: Negative';
        resultDiv.style.color = '#e53935'; // Red color for negative sentiment
    } else {
        resultDiv.innerHTML = 'Sentiment: Neutral';
        resultDiv.style.color = '#000'; // Default color for neutral sentiment
    }
}

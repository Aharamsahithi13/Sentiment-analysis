#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
pd.options.display.max_colwidth=1000


# In[ ]:





# In[3]:


import os
for dirname, _, filenames in os.walk("/Users/dhanarahulsainadiminti/Downloads/sent_ana_dataset/Narendra Modi_data.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


df=pd.read_csv("/Users/dhanarahulsainadiminti/Downloads/sent_ana_dataset/Narendra Modi_data.csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


def remove_usernames_links(tweet):
    s2 = re.sub('http://\S+|https://\S+', '', tweet)
    s1=re.sub(r"#[a-zA-Z0-9\\n@_\s]+","",s2)
    return s1 


# In[9]:


def remove_emoji(txt):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', txt)


# In[10]:


custom_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# In[11]:


def TweetCleaning(tweet):
    link_removal=remove_usernames_links(tweet)
    emoji_removal=remove_emoji(link_removal)
    after_stopword_removal=' '.join(word for word in emoji_removal.split()if word not in custom_stopwords)
    return after_stopword_removal


# In[12]:


def calcPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity


def calcSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return 'positive'
    elif tweet == 0 :
        return 'neutral'
    else:
        return 'negative'


# In[13]:


df["CleanedTweet"]=df["Tweet"].apply(TweetCleaning)
df['tPolarity']=df['CleanedTweet'].apply(calcPolarity)
df['tSubjectivity']=df['CleanedTweet'].apply(calcSubjectivity)
df['segmentation']=df['tPolarity'].apply(segmentation)


# In[14]:


df.head(10)


# In[15]:


df.pivot_table(index=['segmentation'],aggfunc={'segmentation':'count'})


# In[16]:


'''consolidated=' '.join(word for word in df ['CleanedTweet'])
wordCloud=WordCloud(width=400,height=200,random_state=20,max_font_size=119).generate(consolidated)

plt.imshow(wordCloud,interpolation='bilinear')
plt.axis('off')
plt.show()'''


# In[17]:


import seaborn as sns
sns.set_style('whitegrid')
sns.scatterplot(data=df,x='tPolarity',y='tSubjectivity',s=100,hue='segmentation')


# In[18]:


sns.countplot(data=df,x='segmentation')


# In[19]:


df.pivot_table(index=['segmentation'],aggfunc={'segmentation':'count'})


# In[20]:


def predict_sentiment(text):
    cleaned_text = TweetCleaning(text)
    polarity = calcPolarity(cleaned_text)
    
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# User input for text
user_text = input("Enter the text for sentiment prediction: ")

# Predict sentiment for user input
predicted_sentiment = predict_sentiment(user_text)
print(f"The predicted sentiment for the input text is: {predicted_sentiment}")


# In[21]:


# Assuming you have a list of texts with known sentiments
test_data = [
    ("I am feeling great today.", "positive"),
    ("This is a terrible situation.", "negative"),
    ("The weather is okay.", "neutral"),
    # More test data...
]

def evaluate_sentiment(predictions, true_labels):
    correct = 0
    total = len(predictions)
    
    for pred, true_label in zip(predictions, true_labels):
        if pred == true_label:
            correct += 1
    
    accuracy = (correct / total) * 100
    return accuracy

predicted_sentiments = [predict_sentiment(text) for text, _ in test_data]
true_sentiments = [label for _, label in test_data]

accuracy = evaluate_sentiment(predicted_sentiments, true_sentiments)
print(f"Accuracy: {accuracy:.2f}%")


# In[44]:


df1 = pd.read_csv("/Users/dhanarahulsainadiminti/Downloads/sent_ana_dataset/Rahul Gandhi_data.csv")


# In[45]:


df1


# In[46]:


df1["CleanedTweet"]=df1["Tweet"].apply(TweetCleaning)
df1['tPolarity']=df1['CleanedTweet'].apply(calcPolarity)
df1['tSubjectivity']=df1['CleanedTweet'].apply(calcSubjectivity)
df1['segmentation']=df1['tPolarity'].apply(segmentation)


# In[47]:


df1


# In[48]:


unwanted = ['Date','User','Tweet','Time','tPolarity','tSubjectivity']
testing_data = df1.drop(unwanted,axis = 1)


# In[49]:


testing_data1 = testing_data.head(10)


# In[50]:


testing_data


# In[51]:


# Load your test dataset file
test_df = testing_data1 # Replace with your file path

# Assuming your CSV file has columns named 'CleanedTweet' for speech and 'segmentation' for sentiment
test_data = list(test_df[['CleanedTweet', 'segmentation']].itertuples(index=False, name=None))

# Function to predict sentiment using TextBlob
def predict_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Use the test_data to predict sentiments
predicted_sentiments = [predict_sentiment(CleanedTweet) for CleanedTweet, _ in test_data]
true_sentiments = [segmentation for CleanedTweet, segmentation in test_data]

# Evaluate accuracy
def evaluate_sentiment(predictions, true_labels):
    correct = sum(pred == true_label for pred, true_label in zip(predictions, true_labels))
    total = len(predictions)
    accuracy = (correct / total) * 100
    return accuracy

accuracy = evaluate_sentiment(predicted_sentiments, true_sentiments)
print(f"Accuracy: {accuracy:.2f}%")


# In[ ]:





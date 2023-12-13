import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

ny_tweets = pd.read_json("new_york.json",lines=True)
print(len(ny_tweets))
print(ny_tweets.columns)
print(ny_tweets.loc[12]["text"])

l_tweets = pd.read_json("london.json", lines=True)
p_tweets = pd.read_json("paris.json", lines=True)
print(len(l_tweets))
print(len(p_tweets))

ny_text = ny_tweets["text"].tolist()
l_text = l_tweets["text"].tolist()
p_text = p_tweets["text"].tolist()

all_tweets = ny_text + l_text + p_text
labels = [0] * len(ny_text) + [1] * len(l_text) + [2] * len(p_text)

#Training and Testing
train_data,test_data,train_labels,test_labels = train_test_split(all_tweets,labels,test_size=0.2,random_state=1)
print(len(train_data))
print(len(test_data))

counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print(train_data[3])
print(test_data[3])

#Evaluate Dataset
classifier = MultinomialNB()
classifier.fit(train_counts,train_labels)
predictions = classifier.predict(test_counts)

print(accuracy_score(test_labels,predictions))
print(confusion_matrix(test_labels,predictions))

tweet = "The Statue of Liberty is beautiful"
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))

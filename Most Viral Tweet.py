import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

all_tweet = pd.read_json("random_tweets.json", lines = True)

print(len(all_tweet))
print(all_tweet.columns)
print(all_tweet.loc[0]['text'])
print(all_tweet.loc[0]['user']['location'])

retweet_median = all_tweet['retweet_count'].median()
print(retweet_median)
all_tweet['is_viral'] = np.where(all_tweet['retweet_count'] >= retweet_median, 1, 0)
print(all_tweet['is_viral'].value_counts())

all_tweet['tweet_length'] = all_tweet.apply(lambda t : len(t['text']), axis = 1)
all_tweet['followers_count'] = all_tweet.apply(lambda t : t['user']['followers_count'], axis = 1)
all_tweet['friends_count'] = all_tweet.apply(lambda t : t['user']['friends_count'], axis = 1)

labels = all_tweet['is_viral']
data = all_tweet[['tweet_length','followers_count','friends_count']]
scaled_data = scale(data, axis = 0)
print(scaled_data[0])

train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size = 0.2, random_state = 1)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(train_data, train_labels)
print(classifier.score(test_data, test_labels))

scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))
    
plt.plot(range(1,200), scores)
plt.show()

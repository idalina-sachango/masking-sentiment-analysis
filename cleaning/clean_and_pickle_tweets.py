import pandas as pd
from token_cleaning import clean_tweets

df = pd.read_csv("../data/dailies/full_tweets.csv")
df['text'] = df.apply(lambda row : clean_tweets(row['text']), axis = 1)
df.to_pickle("../data/dailies/full_tweets_clean.pkl")
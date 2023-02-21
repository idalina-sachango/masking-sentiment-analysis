import pandas as pd
import os
import matplotlib.pylab as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
from datetime import datetime

# Set up directories
home_path = Path(os.getcwd()).parent
data_path = home_path.joinpath("data/dailies/")

tweets = pd.read_csv(data_path.joinpath("full_tweets.csv"))
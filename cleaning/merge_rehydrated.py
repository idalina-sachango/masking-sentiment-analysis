import pandas as pd
import os
from zipfile import ZipFile
from pathlib import Path
from datetime import datetime
from langdetect import detect
from mask_keywords import keywords


def clean_non_english(df):
    '''
    Cleans removes all non english tweets from an input data frame
    '''
    cleaned = []
    for _, row in df.iterrows():
        try:
            lang = detect(row['text'])
            if lang == 'en':
                cleaned.append(row)
        except:
            cleaned.append(row)
    return pd.DataFrame(cleaned)


home_path = Path(os.getcwd()).parent
data_path = home_path.joinpath("data/dailies")
output_path = data_path.joinpath("rehydrated")


# Assumes all the jsons are in a zipped folder called full_rehydrated_tweets within 
# the dailies folder

# COMMENT OUT THESE IF THE JSONS ARE ALREADY IN YOUR DIRECTORY
zipped = data_path.joinpath("full_rehydrated_tweets.zip")

# Extracting the jsons
with ZipFile(zipped, 'r') as zip_ref:
    zip_ref.extractall(data_path)


# looping over all files in the directory
files = Path(output_path).glob('*')

list_of_dfs = []
for file in files:
    df = pd.read_json(file)
    # Pulling out date rather than datetime
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    df = df[['date', 'id', 'text']]
    list_of_dfs.append(df)

# Concatnating the list of subsetted dfs
int_df = pd.concat(list_of_dfs)


# Filter out non-mask tweets
mask_key = "|".join(keywords)
full_df = int_df[int_df['text'].str.contains(mask_key, case = False)==True]

# Cleaning all non-english tweets
full_df = clean_non_english(full_df)

full_df.to_csv(data_path.joinpath("full_tweets.csv"), index = False)

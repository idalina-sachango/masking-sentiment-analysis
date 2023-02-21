from twarc import Twarc2
from keys import client
from pathlib import Path
from rehydrate_module import retrieve_tweets_by_date, rehydrate
import json
import os

# set directories
home_path = Path(__file__).parent.parent
tweet_ids_path = home_path.joinpath("data/dailies/tweet_ids")
rehydrated_path = home_path.joinpath("data/dailies/rehydrated")

target_dates = ["2020-09-30", "2020-10-01", "2022-01-14", "2022-01-15", "2022-01-16", "2022-03-29"]



# import rehydrated tweets
for date in target_dates:
    retrieve_tweets_by_date(date, tweet_ids_path, rehydrated_path)

    # outfile_tsv = str(tweet_ids_path) +'/clean-' + date + '.tsv.gz'
    # os.unlink(outfile_tsv)



from twarc import Twarc2, expansions
import pandas as pd
import wget
import json
from pathlib import Path
from keys import client


# set directories
home_path = Path(__file__).parent.parent
tweet_ids_path = home_path.joinpath("data/dailies/tweet_ids")
rehydrated_path = home_path.joinpath("data/dailies/rehydrated")

def rehydrate(client, tweet_ids, outfile_name):
    '''
    Rehydrate tweets

    Input (tweet_ids): List of Tweet IDs
    Output: A json file of hydrated tweets
    '''
    # The tweet_lookup function allows
    lookup = client.tweet_lookup(tweet_ids = tweet_ids)
    main_results = []

    for _, page in enumerate(lookup):
        # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
        # so we use expansions.flatten to get all the information in a single JSON
        result = expansions.ensure_flattened(page['data'])
        main_results = main_results + result
    
    with open(outfile_name, 'w') as fout:
        json.dump(main_results, fout)



def retrieve_tweets_by_date(target_date, tweet_ids_path, rehydrated_path):
    # target_date = "2021-01-20"
    dataset_URL = "".join(["https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/", target_date, "/", target_date, "_clean-dataset.tsv.gz?raw=true"])

    #Download the dataset (compressed in a GZ format)
    #!wget dataset_URL -O clean-dataset.tsv.gz
    outfile_tsv = str(tweet_ids_path) +'/clean-' + target_date + '.tsv.gz'
    wget.download(dataset_URL, out = outfile_tsv)

    # gets list of Tweet IDs to lookup; filter to English tweets only
    df = pd.read_csv(outfile_tsv, sep='\t')
    if "lang" in df.columns:
        df = df[df.lang == 'en']
    df['tweet_id'] = df['tweet_id'].astype(str)

    outfile_name = str(rehydrated_path) + '/rehydrated-' + target_date + '.json'
    tweet_ids = list(df['tweet_id'][0:100000]) #subset to 5 for the time being

    return rehydrate(client, tweet_ids, outfile_name)


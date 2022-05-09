# Combines the cleaned listener data, listener responses, and speaker data into
# single csv files for easy input to models

import pandas as pd
import numpy as np
from tqdm import tqdm

# Joins demographic data on listeners with listener responses
listener_data = pd.read_csv("listener_data_numeric.csv")
listener_responses = pd.read_csv("listener_responses_cleaned.csv")
listener_data = pd.merge(listener_data, listener_responses, on="LISTENER_ID")

# Merges listener data withh speaker data, for all sentences
for s in tqdm(range(1, 26)):
    # Reads in acoustic data for speakers and filters out any features for which
    # the feature extraction script was not able to extract a measurement for
    # one or more speakers
    speaker_data = pd.read_csv(f"s{s}/meas.csv")
    speaker_meas = np.asarray(speaker_data.filter(regex=r"F_.+"))
    keep_inds = list(np.logical_not(np.isnan(speaker_meas).any(axis=0)))
    speaker_df = pd.DataFrame(speaker_meas, columns=speaker_data.columns[1:])
    speaker_df = speaker_df.loc[:, keep_inds]
    speaker_df["SPEAKER_ID"] = speaker_data["SPEAKER_ID"]

    # Merges listener and speaker data and writes out to file
    listener_data_s = listener_data[listener_data["SENTENCE"] == s]
    df = pd.merge(speaker_df, listener_data_s, on="SPEAKER_ID")
    df = df.loc[:, (df != 0).any(axis=0)]
    df.to_csv(f"s{s}/data.csv", index=False)

    # Aggregates average ratings per speaker for this sentence and writes out
    avg_df = listener_responses[listener_responses["SENTENCE"] == s]
    avg_df = avg_df.filter(items=["SPEAKER_ID", "RATING"])
    avg_df = avg_df.groupby("SPEAKER_ID").mean()
    avg_df = pd.merge(speaker_df, avg_df, on="SPEAKER_ID")
    avg_df.to_csv(f"s{s}/data_avg.csv", index=False)
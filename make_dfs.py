import pandas as pd
import numpy as np
from tqdm import tqdm

feature_re = r"F_.+"

sentence_ids = range(1, 26)

listener_data = pd.read_csv("listener_data_numeric.csv")
listener_responses = pd.read_csv("listener_responses_cleaned.csv")
listener_data = pd.merge(listener_data, listener_responses, on="LISTENER_ID")

for s in tqdm(range(1, 26)):
    speaker_data = pd.read_csv(f"s{s}/meas.csv")
    speaker_meas = np.asarray(speaker_data.filter(regex=feature_re))
    keep_inds = list(np.logical_not(np.isnan(speaker_meas).any(axis=0)))

    speaker_df = pd.DataFrame(speaker_meas, columns=speaker_data.columns[1:])
    speaker_df = speaker_df.loc[:, keep_inds]
    speaker_df["SPEAKER_ID"] = speaker_data["SPEAKER_ID"]

    listener_data_s = listener_data[listener_data["SENTENCE"] == s]

    df = pd.merge(speaker_df, listener_data_s, on="SPEAKER_ID")
    df = df.loc[:, (df != 0).any(axis=0)]
    df.to_csv(f"s{s}/data.csv", index=False)

    avg_df = listener_responses[listener_responses["SENTENCE"] == s]
    avg_df = avg_df.filter(items=["SPEAKER_ID", "RATING"])
    avg_df = avg_df.groupby("SPEAKER_ID").mean()
    avg_df = pd.merge(speaker_df, avg_df, on="SPEAKER_ID")
    avg_df.to_csv(f"s{s}/data_avg.csv", index=False)
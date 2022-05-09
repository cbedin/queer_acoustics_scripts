import pandas as pd
from read_phones import speaker_re, sentence_re
import json

df = pd.read_csv("listener_data.csv",
    names=['HASH', 'EXP', 'LISTENER_ID', 'BLOCK', 'AGE', 'GENDER', 'PRONOUNS',
            'RACE', 'SEXUAL_ORIENTATION', 'LIVE_REGION', 'URBAN', 'ENGLISH_BG',
            'GREW_IN_US', 'GREW_UP_REGION', 'LGBTQ', 'FRIENDS', 'SPACES', 'ACCURACY'])
df.drop(['HASH', 'EXP', 'BLOCK', 'GREW_IN_US', 'ENGLISH_BG'], axis=1, inplace=True)
df["IS_PROLIFIC"] = df["LISTENER_ID"].apply(lambda x: len(x) == 24)
df = df[df["IS_PROLIFIC"]]
df.drop(['IS_PROLIFIC'], axis=1, inplace=True)
ids = sorted(list(df["LISTENER_ID"].unique()))
id_dict = {id:ids.index(id) + 1 for id in ids}
df["LISTENER_ID"] = df["LISTENER_ID"].apply(lambda x: id_dict[x])
df.fillna('none', inplace=True)
df.to_csv("listener_data_cleaned.csv", index=False)

numeric_df = pd.DataFrame()
numeric_df["LISTENER_ID"] = df["LISTENER_ID"]
for feature in df.columns:
    if feature == "LISTENER_ID":
        continue
    for value in df[feature].unique():
        new_feature_name = f"F_{feature}_{value.upper().replace('_', '-')}"
        numeric_df[new_feature_name] = df[feature].apply(lambda x: int(x == value))
numeric_df.to_csv("listener_data_numeric.csv", index=False)

responses_df = pd.read_csv("listener_responses.csv")
responses_df.rename(columns = {"SubjID":"LISTENER_ID", "file1":"FILE", "response":"RATING", "status":"STATUS"}, inplace = True)
responses_df = responses_df.filter(items=["LISTENER_ID", "FILE", "RATING", "STATUS"])
responses_df = responses_df[responses_df["STATUS"] == "OK"]
responses_df["IS_PROLIFIC"] = responses_df["LISTENER_ID"].apply(lambda x: x in id_dict)
responses_df = responses_df[responses_df["IS_PROLIFIC"]]
ids = list(responses_df["LISTENER_ID"].unique())
responses_df["LISTENER_ID"] = responses_df["LISTENER_ID"].apply(lambda x: id_dict[x])
responses_df["SPEAKER_ID"] = responses_df["FILE"].apply(lambda x: int(speaker_re.search(x).group(1)))
responses_df["SENTENCE"] = responses_df["FILE"].apply(lambda x: int(sentence_re.search(x).group(1)))
responses_df.reset_index(inplace=True)
responses_df = responses_df.filter(items=["LISTENER_ID", "SPEAKER_ID", "SENTENCE", "RATING"])

responses_df.to_csv("listener_responses_cleaned.csv", index=False)

with open('listener_ids.json', 'w') as phones_file: 
    phones_file.write(json.dumps(id_dict))
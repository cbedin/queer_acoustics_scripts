import csv
import numpy as np
from sklearn.decomposition import PCA
import os
import json
from read_phones import zip_files, name_of
import pandas as pd

feature_re = r"F_.+"

phones_file = open('phones.json')
phones = json.load(phones_file)
phones_file.close()

fricatives = phones["fricatives"]
sonorants = phones["sonorants"]
write_out_files = {}
csv_writers = {}
transcriptions = {}
for speaker_id, sentence_id, sound, tg in zip_files():
    transcription = [name_of(p) for p in tg[1] if p.mark]
    if sentence_id not in write_out_files:
        if not os.path.exists(f"s{sentence_id}/"):
            os.mkdir(f"s{sentence_id}/")
        write_out_files[sentence_id] = open(f"s{sentence_id}/meas.csv", "w")
        write_out_files[sentence_id].truncate(0)
        csv_writers[sentence_id] = csv.writer(write_out_files[sentence_id])
        transcriptions[sentence_id] = transcription
        measurement_names = ["SPEAKER_ID"]
        for ind, phone in enumerate(tg[1]):
            name = name_of(phone)
            if name in sonorants:
                measurement_names.extend([f"F_{ind}-{name}_{time_step}-{f_name}" for f_name in ["F0", "F1", "F2", "F3"] for time_step in range(1, 4)])
            elif name in fricatives:
                measurement_names.extend([f"F_{ind}-{name}_{time_step}-COG" for time_step in range(1, 4)])
        csv_writers[sentence_id].writerow(measurement_names)
    assert len(transcription) == len(transcriptions[sentence_id]), f"Transcription mismatch: ({speaker_id},{sentence_id}) should be {transcriptions[sentence_id]}, but is {transcription}"
    formant = sound.to_formant_burg()
    pitch = sound.to_pitch()
    spectrum = sound.to_spectrogram()
    measurements = [speaker_id]
    for phone in tg[1]:
        for time_step in range(1, 4):
            name = name_of(phone)
            mid_time = phone.minTime + (phone.maxTime - phone.minTime) / 4 * time_step
            if name in sonorants:
                f0 = pitch.get_value_at_time(mid_time)
                f1 = formant.get_value_at_time(1, mid_time)
                f2 = formant.get_value_at_time(2, mid_time)
                f3 = formant.get_value_at_time(3, mid_time)
                measurements.extend([f0, f1, f2, f3])
            elif name in fricatives:
                slice = spectrum.to_spectrum_slice(mid_time)
                cog = slice.get_center_of_gravity()
                measurements.append(cog)
    csv_writers[sentence_id].writerow(measurements)

for id in write_out_files:
    write_out_files[id].close()

log = False
if log:
    for i in range(1, 26):
        df = pd.read_csv(f"s{i}/meas.csv")
        meas = np.asarray(df.filter(regex=feature_re))
        meas_logged = np.log(meas, where=np.logical_not(np.isnan(meas)))
        meas = np.nan_to_num(meas)
        df_out = pd.DataFrame(meas_logged,
            columns=df.filter(regex=feature_re).columns,
            index=df["ID"])
        df_out.to_csv(f"s{i}/meas_logged.csv")

pca = False
if pca:
    for i in range(1, 26):
        df = pd.read_csv(f"s{i}/meas.csv")
        meas = np.asarray(df.filter(regex=r"\d{1,2}\-[A-Z]{1,2}_\d\-[0-9A-Z]+"))
        meas = np.nan_to_num(meas)
        pca = PCA()
        components = pca.fit_transform(meas)
        df_out = pd.DataFrame(components,
            columns=[f"C_{i}" for i in range(1, 1 + components.shape[1])],
            index=df["ID"])
        df_out.to_csv(f"s{i}/pca_meas.csv")
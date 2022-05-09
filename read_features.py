# Generates tables of features per speaker per sentence based on the existing
# wav and tg files

import csv
import os
import json
from file_functions import *

# Loads in dict of all the phones
phones_file = open('phones.json')
phones = json.load(phones_file)
phones_file.close()

fricatives = phones["fricatives"]
sonorants = phones["sonorants"]
write_out_files = {}
csv_writers = {}
transcriptions = {}

# Iterates over all speaker/sentence combinations, and generates a feature
# table for each combination
for speaker_id, sentence_id, sound, tg in zip_files():
    transcription = [name_of(p) for p in tg[1] if p.mark]

    # Generates new directory and file if this is the first data we've
    # encountered for this sentence
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
                measurement_names.extend([f"F_{ind}-{name}_{time_step}-{f_name}"
                    for f_name in ["F0", "F1", "F2", "F3"]
                    for time_step in range(1, 4)])
            elif name in fricatives:
                measurement_names.extend([f"F_{ind}-{name}_{time_step}-COG"
                    for time_step in range(1, 4)])
        csv_writers[sentence_id].writerow(measurement_names)
    
    # All sentences must have the same number of phones in their transcription
    # to create a valid table---if this assert is triggered due to a mismatch in
    # transcription, I manually edited the offering file to get the
    # transcriptioins to match
    assert len(transcription) == len(transcriptions[sentence_id])

    # Uses Parselmouth scripts to extract acoustic measurements for all segments
    # in a transcription
    formant = sound.to_formant_burg()
    pitch = sound.to_pitch()
    spectrum = sound.to_spectrogram()
    measurements = [speaker_id]
    for phone in tg[1]:
        for time_step in range(1, 4):
            name = name_of(phone)
            mid_time = phone.minTime + \
                (phone.maxTime - phone.minTime) / 4 * time_step
            if name in sonorants:
                # Extracts pitch and formants for sonorants
                f0 = pitch.get_value_at_time(mid_time)
                f1 = formant.get_value_at_time(1, mid_time)
                f2 = formant.get_value_at_time(2, mid_time)
                f3 = formant.get_value_at_time(3, mid_time)
                measurements.extend([f0, f1, f2, f3])
            elif name in fricatives:
                # Extracts center of gravity for fricatives
                slice = spectrum.to_spectrum_slice(mid_time)
                cog = slice.get_center_of_gravity()
                measurements.append(cog)
    csv_writers[sentence_id].writerow(measurements)

for id in write_out_files:
    write_out_files[id].close()
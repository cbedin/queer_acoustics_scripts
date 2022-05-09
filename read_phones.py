import glob
from parselmouth import Sound
from textgrid import TextGrid
import re
import json

prefix_re = re.compile(r'sp\d{5}_s\d{1,3}_r\d')
speaker_re = re.compile(r'sp(\d{5})')
sentence_re = re.compile(r's(\d{1,2})')

def name_of(phone):
    name = phone.mark
    if "0" in name or "1" in name or "2" in name:
        return name[:-1]
    return name

def zip_files(path="./speaker_data/"):
    wav_files = glob.glob(path + "*.wav")
    tg_files = glob.glob(path + "*.TextGrid")
    wav_files.sort()
    tg_files.sort()

    wav_prefixes = list(map(lambda x: prefix_re.search(x).group(), wav_files))
    tg_prefixes = list(map(lambda x: prefix_re.search(x).group(), tg_files))
    for w, t in zip(wav_prefixes, tg_prefixes):
        assert w == t, f"{w}.wav and {t}.TextGrid do not match! Something's missing."

    speaker_ids = map(lambda x: int(speaker_re.search(x).group(1)), wav_prefixes)
    sentence_ids = map(lambda x: int(sentence_re.search(x).group(1)), wav_prefixes)

    wavs = map(lambda w: Sound(w), wav_files)
    tgs = map(lambda t: TextGrid.fromFile(t), tg_files)
    yield from zip(speaker_ids, sentence_ids, wavs, tgs)

all_phones = set(sum([[name_of(phone) for phone in tg[1]] for _, _, _, tg in zip_files()], start=[]))
all_phones.remove('')

vowels = set([p for p in all_phones if any([v in p for v in "AEIOU"])])
fricatives = set([p for p in all_phones if any([f in p for f in "SZHFV"])]).difference(vowels)
liquids = set([p for p in all_phones if any([l == p for l in "RLWY"])])
nasals = set([p for p in all_phones if any([n in p for n in "MN"])])
stops = set([p for p in all_phones if any([t == p for t in "PTKBDG"])])
sonorants = liquids.union(nasals).union(vowels)

test = vowels.union(fricatives).union(liquids).union(nasals).union(stops)
assert all_phones == test, f"Missed something: {all_phones.difference(test)}"

write_out = {"all_phones": list(all_phones),
            "vowels": list(vowels),
            "fricatives": list(fricatives),
            "liquids": list(liquids),
            "nasals": list(nasals),
            "stops": list(stops),
            "sonorants": list(sonorants)}

with open('phones.json', 'w') as phones_file: 
    phones_file.write(json.dumps(write_out))

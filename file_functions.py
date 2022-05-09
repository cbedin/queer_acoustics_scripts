# Defines some helpful functions and expressions for working with the .wav and
# .TextGrid files

import glob
from parselmouth import Sound
from textgrid import TextGrid
import re

# Regexs useful for extracting the filename, speaker id, or sentence id
prefix_re = re.compile(r'sp\d{5}_s\d{1,3}_r\d')
speaker_re = re.compile(r'sp(\d{5})')
sentence_re = re.compile(r's(\d{1,2})')

def name_of(phone):
    """
    Extracts the name of PHONE, ignoring stress marking for vowels
    """
    name = phone.mark
    if "0" in name or "1" in name or "2" in name:
        return name[:-1]
    return name

def zip_files(path="./speaker_data/"):
    """
    Generator object that yields as a tuple:
        - The speaker ID as a number
        - The sentence ID as a number
        - The .wav file as a sound object
        - The .TextGrid file as a TextGrid object
    For all .wav/.TextGrid file pairs in PATH.

    This is a convenience function that allows us to mass-read and organize all
    the information from all the .wav and .TextGrid files in PATH.
    """

    # Reads in all .wav and .TextGrid files
    wav_files = glob.glob(path + "*.wav")
    tg_files = glob.glob(path + "*.TextGrid")
    wav_files.sort()
    tg_files.sort()

    # Confirms that every .wav file has a matching .TextGrid
    wav_prefixes = list(map(lambda x: prefix_re.search(x).group(), wav_files))
    tg_prefixes = list(map(lambda x: prefix_re.search(x).group(), tg_files))
    for w, t in zip(wav_prefixes, tg_prefixes):
        assert w == t, f"{w}.wav and {t}.TextGrid do not match! \
            Something's missing."

    # Extracts the speaker and sentence ids as lists
    speaker_ids = map(lambda x: int(speaker_re.search(x).group(1)),
        wav_prefixes)
    sentence_ids = map(lambda x: int(sentence_re.search(x).group(1)),
        wav_prefixes)

    # Converts the .wav and .TextGrid files to the respective objects, and then
    # iteratively yields the corresponding information
    wavs = map(lambda w: Sound(w), wav_files)
    tgs = map(lambda t: TextGrid.fromFile(t), tg_files)
    yield from zip(speaker_ids, sentence_ids, wavs, tgs)
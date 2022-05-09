# Generates a complete set of all the phones across all transcriptions, for use
# in feature extraction

import sys
import json
from read_files import *

# Extracts all the phones as a set
all_phones = set([name_of(phone) for _, _, _, tg in zip_files()
    for phone in tg[1]])
all_phones.remove('')

# Breaks the phones up into natural classes
vowels = set([p for p in all_phones if any([v in p for v in "AEIOU"])])
fricatives = set([p for p in all_phones
    if any([f in p for f in "SZHFV"])]).difference(vowels)
liquids = set([p for p in all_phones if any([l == p for l in "RLWY"])])
nasals = set([p for p in all_phones if any([n in p for n in "MN"])])
stops = set([p for p in all_phones if any([t == p for t in "PTKBDG"])])
sonorants = liquids.union(nasals).union(vowels)

# Confirms that all phonemes found in the data were sorted into at least one
# natural class
test = vowels.union(fricatives).union(liquids).union(nasals).union(stops)
assert all_phones == test, f"Missed something: {all_phones.difference(test)}"

# Writes out all the phoneme and natural class information as a .json file for
# later use
write_out = {"all_phones": list(all_phones),
            "vowels": list(vowels),
            "fricatives": list(fricatives),
            "liquids": list(liquids),
            "nasals": list(nasals),
            "stops": list(stops),
            "sonorants": list(sonorants)}
with open('phones.json', 'w') as phones_file: 
    phones_file.write(json.dumps(write_out))

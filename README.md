# Scripts for Applying Machine Learning to Sociophonetic Analysis

This is a collection of scripts I used for analysis purposes in my undergraduate
honors thesis, "(A Computational Approach to) Acoustic Cues of Queer Speech".
These scripts are specific to my project, but can be tweaked or used in
part for similar experiments.

In this project, an experiment was run where a group of speakers were asked to
record themselves reading aloud a list of prescribed sentences from the MOCHA
corpus. Then, listeners were given a task where they listened to a mixed list of
speaker readings and rated each sentence numerically 1--7 for how queer/gay the
speaker sounded to them. This repository contains several scripts for analyzing
the results from this experiment, particularly by applying linear
feature-selection models to the data to look at how a model could decide which
acoustic properties of the speakers' voices may be being used by listeners when
they make their judgements.

## Contents

The following is a list of the all files contained in this repository:

* `read_files.py` - Defines helpful functions and expressions for working with
the `.wav` and `.TextGrid` files, for use in read_phones and read_features

* `read_phones.py` - Generates a complete set of all the phones across all
transcriptions, for use in feature extraction

* `read_features.py` - Generates tables of features per speaker per sentence
based on the existing `.wav` and `.TextGrid` files

* `read_listener_data.py` - Cleans up the raw listener data

* `make_model_inputs.py` - Combines the cleaned listener data, listener
responses, and speaker data into single csv files for easy input to models

* `run_models.py` - Runs linear models on our data, and writes out information
about the results

* `analyze_listener_data.py` - Generates visualizations of the listener data,
without acoustic data on speakers or any of the information from running models

* `analyze_model_data.py` - Generates visualizations of the model outputs, for
the purposes of comparing quality of fit

## Using the scripts

First, create a virtual environment and install the necessary packages.

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The scripts in this repository read data from three sources:

* A local directory containing `.wav` and corresponding `.TextGrid` files for all
speaker/sentence combinations

* A `.csv` file containing survey responses of demographic listener information

* A `.csv` file containing ratings from the listener task

Paths to these sources are included as command-line arguments to any of the
`read` scripts, so we would run any of the read scripts as below:

```
python3 read_phones.py PATH_TO_SPEAKER_DIRECTORY
```

```
python3 read_features.py PATH_TO_SPEAKER_DIRECTORY
```

```
python3 read_listener_data.py PATH_TO_DEMOGRAPHIC_DATA PATH_TO_RESPONSE_DATA
```

All other scripts can be run as-is. The most important thing to note is that
these scripts should be run *in the order seen in the first section*—for the
most part, each script will generate files that are necessary for the next
script to run. These components were all modularized for ease of tweaking
various parts of the pipeline.
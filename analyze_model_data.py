# Generates some visualizations of the model outputs, for the purposes of
# comparing quality of fit

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def run_for(sentence_id, ext="_lasso"):
    """
    Generates plots for sentence SENTENCE_ID, where EXT is a tag to add to
    disambiguate file names---EXT applies to both input and output files.
    """
    preds_df = pd.read_csv(f"s{sentence_id}/predictions{ext}.csv")

    # Plots predicted vs actual ratings for the sentence, and draws
    # a best fit line to compare quality
    plt.rcParams['figure.figsize'] = [7, 7]
    preds_df.plot.scatter(x="RATING", y="PREDICTION", color="green")
    z = np.polyfit(preds_df["RATING"], preds_df["PREDICTION"], 1)
    p = np.poly1d(z)
    plt.plot(range(0, 9), p(range(0, 9)), color="black")
    plt.xlim(1, 7)
    plt.ylim(1, 7)
    plt.xlabel("Actual rating")
    plt.ylabel("Rating predicted by model")
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"s{sentence_id}/a_vs_m{ext}.png")
    plt.close()

# Generates plots for all sentences and methods tested in the paper
for i in tqdm(range(1, 26)):
    run_for(i, ext='_lasso')
    run_for(i, ext='_kbest')
    run_for(i, ext='_lasso_avg')
    run_for(i, ext='_kbest_avg')
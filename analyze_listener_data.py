# Generates some visualizations of the listener data, without acoustic data on
# speakers or any of the information from running models

import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_by(df, group_by, y_label):
    """
    Convenience function that plots the data in DF when sorted by GROUP_BY (we
    either group by the speaker, the listener, or the sentence). Y_LABEL is the
    label to apply to the y-axis.
    """

    # Calculate means and standard deviations for the data
    means = df.groupby([group_by])['RATING'].mean().reset_index()
    means.rename(columns={'RATING':'MEAN_RATING'}, inplace=True)
    stds = df.groupby([group_by])['RATING'].std().reset_index()
    stds.rename(columns={'RATING':'STD_RATING'}, inplace=True)
    stats = pd.merge(means, stds, on=group_by)
    stats.sort_values(by='MEAN_RATING', inplace=True, ascending=False)

    # Plot the means as dots and standard deviations as whiskers
    plt.rcParams['figure.figsize'] = [6, 5]
    plt.errorbar(stats['MEAN_RATING'], range(len(stats)),
            xerr=stats['STD_RATING'], fmt='o', color='green', 
            ecolor='green', elinewidth=2, capsize=2)
    plt.yticks(ticks=range(len(stats)), labels=stats[group_by])
    plt.xticks(ticks=range(1, 8))
    plt.grid(color='lightgray')
    plt.xlabel("Rating given by listeners")
    plt.ylabel(y_label)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout()

# Input path to the csv you want to read as an argument
df = pd.read_csv(sys.argv[1])

plot_by(df, 'SPEAKER_ID', "Speaker ID")
plt.savefig("ratings_by_speaker.png")
plt.close()

plot_by(df, 'LISTENER_ID', "Listener ID")
plt.savefig("ratings_by_listener.png")
plt.close()

plot_by(df, 'SENTENCE', "Sentence ID")
plt.savefig("ratings_by_sentence.png")
plt.close()
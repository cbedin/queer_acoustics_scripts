from tkinter import VERTICAL
import pandas as pd
import matplotlib.pyplot as plt

def plot_by(df, group_by, y_label):
    means = df.groupby([group_by])['RATING'].mean().reset_index()
    means.rename(columns={'RATING':'MEAN_RATING'}, inplace=True)
    stds = df.groupby([group_by])['RATING'].std().reset_index()
    stds.rename(columns={'RATING':'STD_RATING'}, inplace=True)
    stats = pd.merge(means, stds, on=group_by)
    stats.sort_values(by='MEAN_RATING', inplace=True, ascending=False)

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

df = pd.read_csv("listener_responses_cleaned.csv")

plot_by(df, 'SPEAKER_ID', "Speaker ID")
plt.savefig("ratings_by_speaker.png")
plt.close()

plot_by(df, 'LISTENER_ID', "Listener ID")
plt.savefig("ratings_by_listener.png")
plt.close()

plot_by(df, 'SENTENCE', "Sentence ID")
plt.savefig("ratings_by_sentence.png")
plt.close()
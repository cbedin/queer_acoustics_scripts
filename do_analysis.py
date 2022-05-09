import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import r_regression, f_regression
from tqdm import tqdm

feature_re = r"F_.+"

def run_for(sentence_id, ext="_lasso"):

    preds_df = pd.read_csv(f"s{sentence_id}/predictions{ext}.csv")

    plt.rcParams['figure.figsize'] = [7, 7]
    preds_df.plot.scatter(x="RATING", y="PREDICTION", color="green")
    # plt.grid(color='lightgray')
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

    if False:
        pred_data = pd.read_csv(f"s{sentence_id}/predictions{model_ext}{logged}.csv")
        actual = pred_data[pred_data["ALPHA"] == "RATING"]
        if not lasso:
            actual = actual.drop(["K", "ALPHA"], axis=1).transpose().reset_index()
        else:
            actual = actual.drop("ALPHA", axis=1).transpose().reset_index()
        actual.columns = ["PREDICTION_ID", "RATING"]
        pred_data = pred_data[pred_data["ALPHA"] != "RATING"]
        pred_data["ALPHA"] = pred_data["ALPHA"].apply(float)
        if not lasso:
            pred_data["K"] = pred_data["K"].apply(float)
            pred_data = pred_data[(pred_data["K"] == k) & (pred_data["ALPHA"] == alpha)]
            pred_data = pred_data.drop(["K", "ALPHA"], axis=1).transpose().reset_index()
        else:
            pred_data = pred_data[pred_data["ALPHA"] == alpha]
            pred_data = pred_data.drop("ALPHA", axis=1).transpose().reset_index()
        pred_data.columns = ["PREDICTION_ID", "PREDICTION"]
        pred_data = pd.merge(actual, pred_data, on="PREDICTION_ID")
        plt.rcParams['figure.figsize'] = [7, 7]
        pred_data.plot.scatter(x="RATING", y="PREDICTION", color="green")
        # plt.grid(color='lightgray')
        z = np.polyfit(pred_data["RATING"], pred_data["PREDICTION"], 1)
        p = np.poly1d(z)
        plt.plot(range(0, 9), p(range(0, 9)), color="black")
        plt.xlim(1, 7)
        plt.ylim(1, 7)
        plt.xlabel("Actual rating")
        plt.ylabel("Rating predicted by model")
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"s{sentence_id}/a_vs_m{model_ext}{logged}.png")
        plt.close()

    """
    plt.clf()
    plt.figure(figsize = (15,8))
    features = data.sort_values(by='K').filter(regex=feature_re)
    features = features.drop_duplicates()
    sns.heatmap(features, cmap="PiYG", linewidths=0, annot=False, center=0,
        xticklabels=features.columns, yticklabels=False, vmin=-1, vmax=1, cbar=False)
    plt.savefig(f"s{sentence_id}/{pca}k_heatmap.png")
    """

    if False:
        plt.clf()
        plt.figure(figsize = (20,8))
        components = pd.read_csv(f"s{sentence_id}/{pca}components.csv")
        c_names = components["COMPONENT"]
        all_components = components.drop("COMPONENT", 1)
        sns.heatmap(all_components, cmap="PiYG", linewidths=0, annot=False, center=0,
            xticklabels=all_components.columns, yticklabels=c_names, cbar=False)
        plt.savefig(f"s{sentence_id}/{pca}component_heatmap.png")

        plt.clf()
        feature_weights = feature_weights.rename(columns={'FEATURE':'COMPONENT'})
        components_with_weights = pd.merge(components, feature_weights, on="COMPONENT")
        components_with_weights = components_with_weights.drop("COMPONENT", 1).sort_values(by="WEIGHT", ascending=False, key=np.abs)
        components_with_weights = components_with_weights.head(int(len(components_with_weights.index) / 3))
        amnts_of_feats = components_with_weights.apply(np.abs).apply(np.max, axis=0).reset_index()
        amnts_of_feats.columns = ["FEATURE", "MASS"]
        amnts_of_feats = amnts_of_feats[amnts_of_feats["MASS"] > 0.2]
        components_with_weights = components_with_weights.filter(items=amnts_of_feats["FEATURE"])
        w_names = components_with_weights["WEIGHT"]
        components_with_weights = components_with_weights.drop("WEIGHT", 1)
        sns.heatmap(components_with_weights, cmap="PiYG", linewidths=0, annot=False, center=0,
            xticklabels=components_with_weights.columns, yticklabels=w_names, cbar=False)
        plt.savefig(f"s{sentence_id}/{pca}component_weights_heatmap.png")

for i in tqdm(range(1, 26)):
    # run_for(i, forest=True)
    run_for(i, ext='_lasso')
    run_for(i, ext='_kbest')
    run_for(i, ext='_lasso_avg')
    run_for(i, ext='_kbest_avg')
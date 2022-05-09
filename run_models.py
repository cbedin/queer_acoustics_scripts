# Runs linear models on our data, and writes out information about the results

import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, VarianceThreshold, \
    f_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

def run_linear_model(sentence_id, pipeline, parameters, df, ext, cv=None):
    """
    Runs a linear model on the speaker/listener data for a sentence and writes
    out the results.

    Parameters:
        - SENTENCE_ID: ID of the sentence we're working on
        - PIPELINE: Model that we're running, instantiated as an sklearn
            Pipeline object
        - PARAMETERS: Dictionary of the hyperparameters we'll be tuning for this
            model, for use by GridSearchCV
        - DF: Pandas DataFrame containing the data we'll run our model on
        - EXT: String that defines (1) what type of model we're running and (2)
            what the name of the output files will look like
        - CV: Defines what kind of cross-validation we'll be using for use by
            GridSearch CV (defaults to None for five-fold)
    """
    # Separates the feature and rating information, and instantiates and fits a
    # tuned model to the data
    data = df.filter(like="F_")
    ratings = df["RATING"]
    if 'kbest' in ext:
        # If we're using the greedy algorithm then the upper limit of features
        # we're allowed to test will depend on which sentence we're looking at
        parameters['preprocess__k'] = range(1, data.columns.size + 1)
    clf = GridSearchCV(pipeline, parameters, scoring='r2', cv=cv)
    clf.fit(data, ratings)

    # Writes out the ratings predicted by the model, compared to actual ratings
    if "LISTENER_ID" in df.columns:
        predcols = ["SPEAKER_ID", "LISTENER_ID", "RATING"]
    else:
        predcols = ["SPEAKER_ID", "RATING"]
    preds_df = df.filter(items=predcols)
    preds_df["PREDICTION"] = clf.predict(data)
    preds_df.sort_values(by="PREDICTION", inplace=True)
    preds_df.to_csv(f"s{sentence_id}/predictions{ext}.csv", index=False)

    # Writes out the weights the model assigned to each feature
    feature_names = clf.best_estimator_[:-1].get_feature_names_out()
    weights = clf.best_estimator_.named_steps['model'].coef_
    weights_df = pd.DataFrame(np.vstack((feature_names, weights)).T, \
        columns=["FEATURE", "WEIGHT"])
    weights_df["FEATURE"] = weights_df["FEATURE"].apply(lambda x: x[2:])
    weights_df = weights_df[weights_df["WEIGHT"] != 0]
    weights_df.sort_values(by="WEIGHT", inplace=True)
    weights_df.to_csv(f"s{sentence_id}/weights{ext}.csv", index=False)

    # Writes out the cross-validation information from GridSearchCV
    folds_df = pd.DataFrame(clf.cv_results_)
    folds_df = folds_df[folds_df['rank_test_score'] == \
        folds_df['rank_test_score'].min()]
    folds_df = folds_df.transpose().reset_index()
    folds_df.to_csv(f"s{sentence_id}/folds{ext}.csv")

# Reads in information for all sentences
sentence_ids = range(1, 26)
df_by_sentence = {s:pd.read_csv(f"s{s}/data.csv") for s in sentence_ids}
df_by_sentence_avg = {s:pd.read_csv(f"s{s}/data_avg.csv") for s in sentence_ids}

# Defines pipeline and parameter objects for our two linear models
kbest_pipeline = Pipeline(steps=[
    ('drop', VarianceThreshold()),
    ('preprocess', SelectKBest(f_regression)),
    ('scale', MinMaxScaler()),
    ('model', linear_model.RidgeCV())
])
kbest_parameters = {
    'preprocess__k':range(1, 11),
    'model__fit_intercept':[True, False]
}

lasso_pipeline = Pipeline(steps=[
    ('drop', VarianceThreshold()),
    ('scale', MinMaxScaler()),
    ('model', linear_model.LassoCV())
])
lasso_parameters = {
    'model__fit_intercept':[True, False]
}

# Runs our models on both kinds of data x both kinds of models, for all
# sentences
for s in tqdm(sentence_ids):
    run_linear_model(s, kbest_pipeline, kbest_parameters, \
        df_by_sentence[s], '_kbest')
    run_linear_model(s, lasso_pipeline, lasso_parameters, \
        df_by_sentence[s], '_lasso')
    run_linear_model(s, kbest_pipeline, kbest_parameters, \
        df_by_sentence_avg[s], ext='_kbest_avg', cv=3)
    run_linear_model(s, lasso_pipeline, lasso_parameters, \
        df_by_sentence_avg[s], ext='_lasso_avg', cv=3)
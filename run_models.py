import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

def run_linear_model(sentence_id, pipeline, parameters, df, ext, cv=None):
    data = df.filter(like="F_")
    ratings = df["RATING"]
    if 'kbest' in ext:
        parameters['preprocess__k'] = range(1, data.columns.size + 1)
    clf = GridSearchCV(pipeline, parameters, scoring='r2', cv=cv)
    clf.fit(data, ratings)

    if "LISTENER_ID" in df.columns:
        predcols = ["SPEAKER_ID", "LISTENER_ID", "RATING"]
    else:
        predcols = ["SPEAKER_ID", "RATING"]
    preds_df = df.filter(items=predcols)
    preds_df["PREDICTION"] = clf.predict(data)
    preds_df.sort_values(by="PREDICTION", inplace=True)
    preds_df.to_csv(f"s{sentence_id}/predictions{ext}.csv", index=False)

    feature_names = clf.best_estimator_[:-1].get_feature_names_out()
    weights = clf.best_estimator_.named_steps['model'].coef_
    weights_df = pd.DataFrame(np.vstack((feature_names, weights)).T, columns=["FEATURE", "WEIGHT"])
    weights_df["FEATURE"] = weights_df["FEATURE"].apply(lambda x: x[2:])
    weights_df = weights_df[weights_df["WEIGHT"] != 0]
    weights_df.sort_values(by="WEIGHT", inplace=True)
    weights_df.to_csv(f"s{sentence_id}/weights{ext}.csv", index=False)

    folds_df = pd.DataFrame(clf.cv_results_)
    folds_df = folds_df[folds_df['rank_test_score'] == folds_df['rank_test_score'].min()]
    folds_df = folds_df.transpose().reset_index()
    folds_df.to_csv(f"s{sentence_id}/folds{ext}.csv")

def run_forest_model(sentence_id, ratings, data):
    best_score = -float('inf')
    best_d = 0
    for d in tqdm(range(1, int(data.shape[1] / 5) + 1)):
        cv_results = cross_validate(RandomForestRegressor(max_depth=d),
                                    data, ratings, scoring="neg_mean_squared_error")
        score = sum(cv_results['test_score'])
        if score > best_score:
            best_score = score
            best_d = d

    clf = RandomForestRegressor(max_depth=best_d)
    clf.fit(data, ratings)

    predictions = clf.predict(data)
    preds_df = pd.DataFrame(np.vstack((ratings, predictions)).T, columns=["RATING", "PREDICTION"])
    preds_df["PREDICTION"] = preds_df["PREDICTION"].apply(lambda x: round(x, 2))
    preds_df.to_csv(f"s{sentence_id}/predictions_forest.csv", index=False)

    cv_results = cross_validate(clf, data, ratings, scoring="r2")
    folds_df = pd.DataFrame(cv_results['test_score'], columns=["VALUE"], index=[f"FOLD {i}" for i in range(1, 6)])
    folds_df = folds_df.append(pd.DataFrame([best_d, clf.score(data, ratings)],
                                            columns=["VALUE"], index=["MAX_DEPTH", "OVERALL"]))
    folds_df["VALUE"] = folds_df["VALUE"].apply(lambda x: round(x, 2))
    folds_df.to_csv(f"s{sentence_id}/folds_forest.csv", index=False)

sentence_ids = range(1, 26)
df_by_sentence = {s:pd.read_csv(f"s{s}/data.csv") for s in sentence_ids}
df_by_sentence_avg = {s:pd.read_csv(f"s{s}/data_avg.csv") for s in sentence_ids}

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

for s in tqdm(sentence_ids):
    run_linear_model(s, kbest_pipeline, kbest_parameters, df_by_sentence[s], '_kbest')
    run_linear_model(s, lasso_pipeline, lasso_parameters, df_by_sentence[s], '_lasso')
    run_linear_model(s, kbest_pipeline, kbest_parameters, df_by_sentence_avg[s], ext='_kbest_avg', cv=3)
    run_linear_model(s, lasso_pipeline, lasso_parameters, df_by_sentence_avg[s], ext='_lasso_avg', cv=3)
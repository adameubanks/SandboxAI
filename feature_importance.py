from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

def k_best(X,y):
    selected_features = SelectKBest(score_func=chi2, k=10)
    fit = selected_features.fit(X,y)
    df_scores = pd.DataFrame(fit.scores_)
    df_cols = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_cols,df_scores],axis=1)
    feature_scores.columns = ['Feature','Score']
    return feature_scores.nlargest(10,'Score')
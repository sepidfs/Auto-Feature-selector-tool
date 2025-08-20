import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

player_df = pd.read_csv(r"C:\Users\14163\Desktop\university cu boulder\GorgeBrown\Mashine Learning 1\Assignment 7\fifa19.csv")


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


player_df = player_df[numcols+catcols]



traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


traindf = pd.DataFrame(traindf,columns=features)


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


X.head()


len(X.columns)


 ### Set some fixed set of features

feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


 ## Filter Feature Selection - Pearson Correlation


def cor_selector(X, y, num_feats):
    cor_list = []
    for i in X.columns:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(abs(cor))  # take absolute value

    cor_list = pd.Series(cor_list, index=X.columns)
    cor_feature = cor_list.sort_values(ascending=False).index[:num_feats].tolist()
    cor_support = [True if col in cor_feature else False for col in X.columns]
    return cor_support, cor_feature


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


 ### List the selected features from Pearson Correlation


cor_feature


 ## Filter Feature Selection - Chi-Sqaure


 ### Chi-Squared Selector function


def chi_squared_selector(X, y, num_feats):
    

    #  Scale the features to be in [0,1] range
    X_norm = MinMaxScaler().fit_transform(X)

    # Apply SelectKBest with chi2 scoring function
    chi_selector = SelectKBest(score_func=chi2, k=num_feats)
    chi_selector.fit(X_norm, y)

    # Step 3: Get the boolean mask and feature names
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()

    # Your code ends here
    return chi_support, chi_feature

chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


### List the selected features from Chi-Square 


chi_feature

 ### RFE Selector function


def rfe_selector(X, y, num_feats):
    # Scale the features
    X_scaled = MinMaxScaler().fit_transform(X)

    # Initialize a base model
    model = LogisticRegression(max_iter=1000)

    # Run RFE with verbose output
    rfe = RFE(estimator=model, n_features_to_select=num_feats, step=1, verbose=1)
    rfe.fit(X_scaled, y)

    # Extract selected features
    rfe_support = rfe.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()

    return rfe_support, rfe_feature

rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


 ### List the selected features from RFE


rfe_feature


 ## Embedded Selection - Lasso: SelectFromModel

def embedded_log_reg_selector(X, y, num_feats):
    # Step 1: Normalize features to [0, 1]
    X_scaled = MinMaxScaler().fit_transform(X)

    # Step 2: Fit Logistic Regression with L1 penalty
    lasso = LogisticRegression(penalty="l1", solver='liblinear', max_iter=1000)
    lasso.fit(X_scaled, y)

    # Step 3: Select top `num_feats` features based on coefficient magnitude
    coef = np.abs(lasso.coef_)[0]
    top_indices = np.argsort(coef)[-num_feats:]

    # Step 4: Create support mask and feature names list
    embedded_lr_support = [i in top_indices for i in range(X.shape[1])]
    embedded_lr_feature = X.columns[top_indices].tolist()

    return embedded_lr_support, embedded_lr_feature


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


embedded_lr_feature


 ## Tree based(Random Forest): SelectFromModel


def embedded_rf_selector(X, y, num_feats):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    selector = SelectFromModel(rf, max_features=num_feats, prefit=True)
    embedded_rf_support = selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature


embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')  
print("Selected features:", embedded_rf_feature)


# Assuming embedded_rf_feature is already defined
print("[")
for feat in embedded_rf_feature:
    print(f"    '{feat}',")
print("]")



  ## Tree based(Light GBM): SelectFromModel


def embedded_lgbm_selector(X, y, num_feats):
    from lightgbm import LGBMClassifier

    # Ensure feature names are valid
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    top_indices = importances.argsort()[-num_feats:]

    embedded_lgbm_support = [i in top_indices for i in range(X.shape[1])]
    embedded_lgbm_feature = X.columns[top_indices].tolist()

    return embedded_lgbm_support, embedded_lgbm_feature


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


embedded_rf_feature  


# put all selection together
feature_selection_df = pd.DataFrame({
    'Feature': feature_name,
    'Pearson': cor_support,
    'Chi-2': chi_support,
    'RFE': rfe_support,
    'Logistics': embedded_lr_support,
    'Random Forest': embedded_rf_support,
  
})

# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df.iloc[:, 1:], axis=1)

# sort and display the top 30
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=[False, True])
feature_selection_df.index = range(1, len(feature_selection_df) + 1)
feature_selection_df.head(30)



# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?


def preprocess_dataset(dataset_path):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Define relevant features
    numcols = ['Overall', 'Crossing','Finishing','ShortPassing','Dribbling','LongPassing',
               'BallControl','Acceleration','SprintSpeed','Agility','Stamina','Volleys',
               'FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots',
               'Aggression','Interceptions']

    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    # Select only those columns
    df = df[numcols + catcols]

    # Drop rows with missing values and reset index
    df = df.dropna().reset_index(drop=True)

    # Binary target based on Overall score
    y = (df['Overall'] >= 87)

    # Prepare numeric and categorical features
    X_num = df[numcols].drop(columns='Overall')
    X_cat = pd.get_dummies(df[catcols], drop_first=True)

    # Normalize numeric features
    scaler = MinMaxScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns).reset_index(drop=True)

    # Reset index for categorical dummies to ensure alignment
    X_cat = X_cat.reset_index(drop=True)

    # Combine scaled numeric and encoded categorical features
    X = pd.concat([X_num_scaled, X_cat], axis=1)

    # Clean column names for LightGBM compatibility
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    num_feats = X.shape[1]

    return X, y, num_feats



def embedded_lgbm_selector(X, y, num_feats):
    # Clean column names to remove unsupported characters
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-num_feats:]

    embedded_lgbm_support = [i in top_indices for i in range(X.shape[1])]
    embedded_lgbm_feature = X.columns[top_indices].tolist()

    return embedded_lgbm_support, embedded_lgbm_feature




def autoFeatureSelector(dataset_path, methods=[], top_k=5):
    

    # Preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    # Store selected features per method
    selected_features_per_method = []

    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
        selected_features_per_method.append(cor_feature)

    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
        selected_features_per_method.append(chi_feature)

    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
        selected_features_per_method.append(rfe_feature)

    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        selected_features_per_method.append(embedded_lr_feature)

    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        selected_features_per_method.append(embedded_rf_feature)

    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        selected_features_per_method.append(embedded_lgbm_feature)

    # Flatten all features selected across methods
    all_features = [feature for sublist in selected_features_per_method for feature in sublist]

    # Count frequency of each feature
    vote_count = Counter(all_features)

    # Sort by vote count (descending), then alphabetically
    sorted_votes = sorted(vote_count.items(), key=lambda x: (-x[1], x[0]))

    # Return top_k features
    best_features = [feature for feature, count in sorted_votes[:top_k]]

    return best_features





best_features = autoFeatureSelector(
    dataset_path=r"C:\Users\14163\Desktop\university cu boulder\GorgeBrown\Mashine Learning 1\Assignment 7\fifa19.csv",
    methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm']
)
best_features



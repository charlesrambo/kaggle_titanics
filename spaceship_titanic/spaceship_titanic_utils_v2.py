# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 15:48:34 2026

@author: cramb
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif, RFECV, SequentialFeatureSelector
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from optuna_integration import OptunaSearchCV

import category_encoders as ce
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict 
from sklearn.metrics import log_loss, accuracy_score


def get_num_bins(num_obs, corr = None):
    # Optimal number of bins for discretization
    
    # Univariate case
    if corr is None:
        
        zeta = np.cbrt(8 + 324 * num_obs + 12 * np.sqrt(36 * num_obs + 729 * num_obs**2))
        
        bins = np.round(zeta/6 + 2/(3 * zeta) + 1/3)
        
    # Bivariate case
    else:
        
        # Clip correlation so the number of bins doesn't explode
        corr = np.clip(corr, -0.95, 0.95)
        
        bins = np.round(np.sqrt(1/2) * np.sqrt(1 + np.sqrt(1 + 24 * num_obs/(1 - corr**2))))
           
        
    return int(np.max([bins, 2]))


def preprocess_for_mi(df):
    """
    Fills, factorizes, and discretizes the entire dataframe once
    using optimized marginal binning.
    """
    
    df_discrete = df.copy()
    
    # Calculate the universal optimal bin count for this sample size
    bins = get_num_bins(len(df_discrete))
    
    for col in df_discrete:
        
        # Handle Categorical / Boolean / Object
        if df_discrete[col].dtype == 'object' or \
           df_discrete[col].dtype.name == 'category' or \
           pd.api.types.is_bool_dtype(df_discrete[col]):
            
            df_discrete[col], _ = df_discrete[col].factorize()
            df_discrete[col] = df_discrete[col].fillna(-1)
            
        # Handle Continuous (Float/Int)
        else:
            
            # Impute with mean
            df_discrete[col] = df_discrete[col].fillna(df_discrete[col].mean())
            
            # Use Rank-based Quantile binning to maximize entropy
            df_discrete[col] = pd.qcut(df_discrete[col].rank(method = 'first'), 
                                       q = bins, labels = False)
            
    return df_discrete


# Create function to get mutual information
def get_nmi(df, x_cols, y_col, random_state = 0, n_jobs = -1):
    """
    Calculates normalized mutual information (NMI) on a discretized data frame.
    """
    X = df[x_cols]
    y = df[y_col]
    
    # Calculate Shannon Entropy of the target
    target_entropy = entropy(y.value_counts(normalize = True))
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, discrete_features = True, 
                                        random_state = random_state, 
                                        n_jobs = n_jobs)
    
    # Create NMI series
    nmi_series = pd.Series(mi_scores/target_entropy, index = x_cols)
    
    return nmi_series


def create_nmi_df(df, n_jobs = -1):
    """
    Systematically calculates the NMI matrix for all columns.
    """
    
    # Discretize the whole data frame once
    df_preped = preprocess_for_mi(df)
    
    # Initialize dictionary to hold results
    nmi_results = {}
    
    # Iterate through targets
    for y_col in df_preped:
        
        x_cols = [col for col in df_preped if col != y_col]
        
        nmi_results[y_col] = get_nmi(df_preped, x_cols, y_col, n_jobs = n_jobs)
    
    # Create DataFrame and fill diagonal with 1.0
    final_df = pd.DataFrame(nmi_results).fillna(1.0)
    
    # Sort the final result
    final_df = final_df.loc[final_df.index, final_df.index]
    
    return final_df


def fill_by_group(df, group_cols, target_col):
    """Fills NaNs in target_col using the mode of the specified groups."""
    
    # Create mapping
    mode_map = df.groupby(group_cols)[target_col].transform(lambda x: x.mode().get(0))
    
    # Fill missing results
    df[target_col] = df[target_col].fillna(mode_map)
    
    return df


def apply_imputation_heuristics(df, amenities):
    
    # Make a copy so we don't change df
    df = df.copy()
    
    # Fill cabin number and deck for families
    for target_col in ['Side', 'Deck', 'CabinBlock']:
        
        df = fill_by_group(df, ['Group', 'LastName'], target_col)
    
    # Then use simply group
    for target_col in ['HomePlanet', 'Destination', 'Side', 'Deck', 'CabinBlock']:
        
        df = fill_by_group(df, 'Group', target_col)
        
    # Fill mising values, if same last name, home planet, and destination
    for target_col in ['Side', 'Deck', 'CabinBlock']:
    
        df = fill_by_group(df, ['LastName', 'HomePlanet', 'Destination'], target_col)
    
    # Assume the rest of the missing side people are port, since this is the minorty and should be about even
    df['Side'] = df['Side'].fillna('P')
    
    # If in cryo sleep or under 13, you didn't spend any money
    mask_no_spend = (df['CryoSleep'] == True) | (df['Age'] < 13)

    # Get whether you spent money
    mask_spend = df[amenities].sum(axis = 1) > 0
    
    # Fill missing median age with global median
    df['MedianAge'] = df['MedianAge'].fillna(df['Age'].median())

    # If CryoSleep is True, then no amenities 
    df.loc[mask_no_spend, amenities] = df.loc[mask_no_spend, amenities].fillna(0)

    # Not in CryoSleep if the sum of the amenitites is greater than 0
    df.loc[mask_spend, 'CryoSleep'] = df.loc[mask_spend, 'CryoSleep'].fillna(False)

    # These next imputations are for people not in cryo sleep
    not_cryosleep = df['CryoSleep'] != True

    for col in ['ShoppingMall', 'RoomService', 'VRDeck']:
        
        # Is there another buyer for the group
        group_spending = df.loc[not_cryosleep].groupby(['Group', 'LastName'])[col].transform('sum')
        
        # If there is then fill with 0, since usually only one buyer per group
        df.loc[not_cryosleep & (group_spending > 0), col] = df.loc[not_cryosleep & (group_spending > 0), col].fillna(0)

    # Only one VIP per group
    df['has_VIP'] = df.groupby('Group')['VIP'].transform('any')

    # When isn't the observation a VIP?
    vip_false_mask = (
        (df['has_VIP'] & df['VIP'].isna()) | 
        ((df['Age'] < 18) & df['VIP'].isna()) | 
        ((df['CryoSleep'] == True) & df['VIP'].isna()))


    # Fill these VIPs observations with false
    df.loc[vip_false_mask, 'VIP'] = False

    # Drop columns we don't need
    df = df.drop(columns = ['has_VIP'], errors = 'ignore')

    return df


def get_best_imputation_model(df, target_col, selected_preds, feature_types, 
                              models_dict, scoring, n_jobs = -1, random_state = 0):
    """
    Core engine to find and tune the best model for a specific target column.
    Shared by both general and Bayesian imputation workflows.
    """
    # Separate Features for preprocessing
    num_features = [f for f in selected_preds if feature_types[f] == 'numerical']
    cat_features = [f for f in selected_preds if feature_types[f] == 'categorical']
    
    # Preprocessing layers
    num_transformer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler())
        ])
    
    cat_transformer = Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'most_frequent')),
            ('encode', ce.TargetEncoder()) 
        ])

    
    preprocessor = ColumnTransformer(transformers = [
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

    # Data Prep & Target Encoding Fix
    X_train = df.loc[df[target_col].notna(), selected_preds]
    y_train = df.loc[df[target_col].notna(), target_col]
    
    # Need to encode the labels for xgboost
    is_numeric = feature_types[target_col] == 'numerical'
    
    # Intialize object to hold label encoder
    le = None
    
    # Label Encode the target if it is categorical 
    if not is_numeric:
        
        le = LabelEncoder()
        
        # Ensure we treat target as string for classification consistency
        y_train = le.fit_transform(y_train.astype(str))
    
    # Get dictionary of available models depending on whether it's regression or classification
    models = models_dict['reg'] if is_numeric else models_dict['clf']
    
    # Initialize objects to hold results
    best_overall_model = None
    best_model_name = None
    
    # Negative infinity is the starting values to beat
    best_overall_score = -np.inf
    
    for name, (model, params) in models.items():
        
        
        pipe = Pipeline([('pre', preprocessor), ('model', model)])
        
        # If it's categorical...
        if not is_numeric:
            
            # ... we want each holdout fold to have the same proportion of each category
            cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)
            
        else: 
            cv = 5
            
        opt = OptunaSearchCV(estimator = pipe,
                             param_distributions = params, 
                             n_trials = 40, 
                             cv = cv,
                             scoring = scoring,
                             random_state = random_state,
                             n_jobs = n_jobs)
        
        opt.fit(X_train, y_train)
        
        if opt.best_score_ > best_overall_score:
            
            best_overall_score = opt.best_score_
            best_overall_model = opt.best_estimator_
            best_model_name = name
            
    return best_overall_model, le, best_overall_score, best_model_name


def run_systematic_imputation(df, target_col, nmi_df, feature_types, models_dict,
                              nmi_threshold = 0.02, n_jobs = -1, random_state = 0):
    
    print(f"\n--- Systematic Imputation: {target_col} ---")
    
    # We allow 'CabinBlock' but forbid the raw 'CabinNum'
    forbidden = ['Group', 'Num', 'FirstName', 'LastName', 'Transported', 'PassengerId']
    
    potential_preds = nmi_df[target_col].drop(labels=[target_col], errors = 'ignore')
    selected_preds = potential_preds[potential_preds > nmi_threshold].index.tolist()
    selected_preds = [f for f in selected_preds if f in df.columns and f not in forbidden]
    
    if not selected_preds:
        
        print(f'Skipping {target_col}: No valid predictors found.')
        
        return df, None, None

    # Specify what type of scoring we're doing
    scoring = 'r2' if feature_types[target_col] == 'numerical' else 'accuracy'
    
    # Use Helper to find best model
    best_model, le, score, name = get_best_imputation_model(df, target_col, selected_preds, 
                                                        feature_types, models_dict, 
                                                        scoring, n_jobs, random_state)
    
    # Standard Prediction Logic
    missing_mask = df[target_col].isna()

    #  Apply to dataframe
    if missing_mask.any():
        
        # Get data frame of missing values
        X_missing = df.loc[missing_mask, selected_preds]
        
        # Make predictions
        preds = best_model.predict(X_missing)
        
        # If le isn't none... 
        if le is not None:
            
            # ... transform predictions back into their originals
            preds = le.inverse_transform(preds)
            
            # Cast back to original expected types for consistency
            if set(preds).issubset({'True', 'False'}):
                
                preds = [p == 'True' for p in preds]
        
        # Fill data frame
        df.loc[missing_mask, target_col] = preds
        
        print(f'Filled {len(X_missing)} NaNs. Best {scoring}: {score:.4f}')
    
    return df, best_model, name


def bayesian_block_imputer(df, nmi_df, feature_types, models_dict, starboard_state,
                           nmi_threshold = 0.02, n_jobs = -1, random_state = 0):
    """
    Nuanced imputer for CabinBlock using:
    Posterior = ML_Likelihood * Physical_Side_Mask * Deck_Capacity_Prior
    """
    
    # Make a deep copy of df
    df = df.copy()
    
    # This is for cabin block imputation
    target_col = 'CabinBlock'
    
    print(f"\n--- Bayesian Systematic Imputation: {target_col} ---")
    
    # Select Predictors
    forbidden = ['Group', 'Num', 'FirstName', 'LastName', 'Transported', 'PassengerId'] 
    
    potential_preds = nmi_df[target_col].drop(labels = [target_col], errors = 'ignore')
    
    selected_preds = potential_preds[potential_preds > nmi_threshold].index.tolist()
    
    selected_preds = [f for f in selected_preds if f in df.columns and f not in forbidden]
    
    # Fit Best Model via Helper
    best_model, le, score, name = get_best_imputation_model(df, target_col, selected_preds, 
                                                        feature_types, models_dict, 
                                                        'accuracy', n_jobs, random_state)
    
    print(f'Using model {name}. Its accuracy was {score:.4f}.')
    
    # Get mask of missing values
    missing_mask = df[target_col].isna()
    
    if not missing_mask.any(): return df
    
    # If there are missing values, get corresponding predictions
    X_missing = df.loc[missing_mask, selected_preds]
    
    # Get Raw likelihoods using the machine learning model
    ml_probs = best_model.predict_proba(X_missing)
    
    # Get classes; make them float so we can use inequalities
    classes = le.classes_.astype(float)
    
    # Get the side and deck
    sides = df.loc[missing_mask, 'Side'].values
    decks = df.loc[missing_mask, 'Deck'].values
    
    # Calculate the counts for each deck
    all_counts = df.groupby(['Deck', 'CabinBlock']).size().to_dict()
    
    # Initialize list to hold predictions
    final_preds = []
    
    for i in range(len(X_missing)):
        
        # Initialize side mask
        side_mask = np.ones_like(ml_probs[i, :])
        
        # Blocks are defined so that less than starboard_state it's port and >= starboard_state it's starboard
        
        # If it's starboard...
        if sides[i] == 'S':
            
            # ... we won't let it be a portside block
            side_mask[classes < starboard_state] = 0
         
        # If it's portside... 
        else:
            
            # ... we won't let it be a starboard side block
            side_mask[classes >= starboard_state] = 0
        
        # Capacity Prior (1 / (count + 1))
        deck = decks[i]
        
        # Our prior is that the probability of being in each block is inversely proportaionl to the number of people in the blcok
        cap_prior = np.array([1.0/(all_counts.get((deck, c), 0) + 1) for c in classes])
        
        # Bayesian Combination
        posterior = ml_probs[i, :] * side_mask * cap_prior
        
        # Pick the Maximum A Posteriori (MAP) estimate
        if posterior.sum() == 0: 
            
            # Fallback if mask kills all options
            final_preds.append(classes[np.argmax(ml_probs[i])])
            
        else:
            
            final_preds.append(classes[np.argmax(posterior)])
            
     # Fill in the values       
    df.loc[missing_mask, target_col] = final_preds
    
    print(f'Filled {len(X_missing)} NaNs using Bayesian Logic. CV Accuracy: {score:.4f}')        
        
    return df


def plot_ship_transport_heatmap(df, starboard_state):
    """
    Creates a geometrically accurate heatmap of transport rates 
    across the ship's physical layout.
    """
    # Make a deep copy of df
    df = df.copy()
    
    # Make sure transported is a float
    df['Transported'] = df['Transported'].astype(float)
    
    # Make cabin block an integer
    df['CabinBlock'] = df['CabinBlock'].astype(int)
    
    # Aggregate transport rate by Deck and CabinBlock
    
    # Calculate the Color Matrix 
    heatmap_data = df.groupby(['Deck', 'CabinBlock'])['Transported'].mean().unstack()
    
    # Calculate the Annotation Matrix (% of total passengers)
    
    # Get counts per block
    count_data = df.groupby(['Deck', 'CabinBlock']).size().unstack()/len(df)

    # Sort the deck so they're in order
    deck_order = sorted(df['Deck'].unique())
    
    # Reorder our data sets
    heatmap_data = heatmap_data.reindex(deck_order)
    count_data = count_data.reindex(deck_order)
    
    # Setup Plot
    plt.figure(figsize = (20, 10))
    
    # We use a diverging or sequential colormap
    sns.heatmap(heatmap_data, 
                fmt = '.2%',
                annot = count_data, 
                cmap = 'coolwarm', 
                linewidths = 0.5,
                annot_kws = {'size': 8},
                cbar_kws = {'label': 'Portion Transported'})

    # Add "Architectural" Annotations
    plt.title('Spaceship Titanic: Transport Rate (Color) & Passenger Density % (Text)', 
              fontsize = 16)
    
    plt.xlabel(f"Cabin Block (Port: 0-{starboard_state - 1:0.0f} | Starboard: {starboard_state:0.0f}-{df['CabinBlock'].max():0.0f}", 
               fontsize = 12)
    
    plt.ylabel('Deck', fontsize = 12)
    
    # Add a vertical line to mark the seam between Port and Starboard
    plt.axvline(x = starboard_state, color = 'black', linestyle='--', 
                linewidth = 2, label = 'Ship Centerline')
    
    plt.tight_layout()
    
    plt.savefig('ship_heatmap.png')
    
    plt.show()
 

def audit_results(X, y, estimator, cv):
    
    # Generate Out-Of-Fold predictions for the training set
    # This gives us a 'Survival Probability' for every person in the training data
    oof_probs = cross_val_predict(estimator, X, y, cv = cv, method = 'predict_proba')[:, 1]

    #  Attach probabilities to the training dataframe for analysis
    audit_df = X.copy()
    audit_df['Actual'] = y
    audit_df['Prob_Transported'] = oof_probs
    audit_df['Error'] = abs(audit_df['Actual'] - audit_df['Prob_Transported'])
    
    # Sort values so biggest errors first
    audit_df = audit_df.sort_values(by = 'Error', ascending = False)

    return audit_df


def plot_confidence_audit(audit_df):
    
    plt.figure(figsize = (12, 7))
    
    # Plot with KDE to show the "shape" of the classes
    sns.histplot(data = audit_df, 
                 x = 'Prob_Transported', 
                 hue = 'Actual', 
                 element = 'step', 
                 bins = 30, 
                 alpha = 0.5, 
                 stat = 'probability',
                 common_norm = False)
    
    # Decision Boundary
    plt.axvline(x = 0.5, color = 'black', linestyle = '--', label = 'Decision Boundary')
    
    plt.title('Probability Distributions: Are We Seperating Classes Well?', 
              fontsize = 14)
    
    plt.xlabel('Model Confidence (Probability of Transported)')
    
    plt.ylabel('Passenger Count')
    
    plt.legend(title = 'Actual Outcome', labels = ['Transported', 'Stayed', 'Boundary'])
    
    plt.savefig('confidence_plots.png')
    
    plt.show()
    
    
def plot_feature_bias_audit(audit_df, train_df, feature, num_bins = 5):
    """
    Plots accuracy bias, automatically binning numerical features 
    to ensure a readable categorical bar chart.
    """
    
    # Make a deep copy of audit df
    temp_df = audit_df.copy()
    
    # Merge the feature from train_df if it's not in audit_df
    if feature not in temp_df.columns:
        
        temp_df = temp_df.merge(train_df[[feature]], left_index = True, 
                                  right_index = True)
    
    # Get the correct predictions
    temp_df['Correct'] = (temp_df['Prob_Transported'] > 0.5) == temp_df['Actual']
    
    display_feature = feature
    
    # Check if the feature is numeric and has many unique values
    if np.issubdtype(temp_df[feature].dtype, np.number) and temp_df[feature].nunique() > 10:
        
        # Create bins so we don't have a bar for every single number
        temp_df[f'{feature}_binned'] = pd.qcut(temp_df[feature], q = num_bins, 
                                               duplicates = 'drop').astype(str)
        
        display_feature = f'{feature}_binned'
    
    # Calculate accuracy per category
    bias = temp_df.groupby(display_feature)['Correct'].mean().sort_values()
    
    plt.figure(figsize = (10, 6))
    
    bias.plot(kind = 'barh')
    
    avg_acc = temp_df['Correct'].mean()
    
    plt.axvline(x = avg_acc, color = 'blue', linestyle = '--', 
                label = f'Global Avg ({avg_acc:.2%})')
    
    plt.title(f'Accuracy Bias Audit: {feature}', fontsize = 14)
    
    plt.xlabel('Accuracy Rate')
    
    plt.ylabel(display_feature)
    
    # Accuracy is 0 to 1
    plt.xlim(0, 1) 
    
    plt.legend()
    
    plt.grid(axis = 'x', alpha = 0.3)
    
    plt.tight_layout()
    
    plt.savefig(f'bias_{feature}.png')
    
    plt.show()


# This hasn't affected the results, but leaving in regardless
def apply_spaceship_group_mask(train_df, test_df, pred_probas, upper = 0.9, 
                               lower = 0.1, threshold = 0.5):
    """
    Adjusts predictions based on the transport status of other group members.
    Uses 'Group' (extracted from PassengerId) as the key.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Add labels
    test_df['Test'], train_df['Test'] = True, False
    
    # Add the probabilites to test_df
    test_df['Transported'] = pred_probas
    
    # Combine the results
    df = pd.concat([train_df, test_df], axis = 0)
    
    # Calculate transport stats per group from Training Data
    group_stats = df.groupby('Group')['Transported'].transform('mean').fillna(0.5)
    
    # Force Transported if group mean was 1
    mask_transported = (group_stats > upper) & (df['GroupSize'] > 1)
    df.loc[mask_transported, 'Transported'] += 0.2
    
    # Force Not Transported if group mean was 0
    mask_stayed = (group_stats < lower) & (df['GroupSize'] > 1)
    df.loc[mask_stayed, 'Transported'] -= 0.2
    
    # Convert back to binary predictions
    final_preds = df.loc[df['Test'], 'Transported'] > threshold
    
    # Subset our masks to just test
    mask_transported = mask_transported.loc[df['Test']]
    mask_stayed = mask_stayed.loc[df['Test']]
    
    transported_overrides = (mask_transported & (pred_probas < threshold)).sum()
    stayed_overrrides = (mask_stayed & (pred_probas > threshold)).sum()
    
    print(f"Group Mask: Overrode {transported_overrides} to Transported and {stayed_overrrides} to Stayed.")
    
    return final_preds

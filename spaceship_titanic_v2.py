# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 17:07:00 2026

@author: cramb
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import gc
import spaceship_titanic_utils_v2 as utils

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier


from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna_integration import OptunaSearchCV
import optuna
from optuna.exceptions import ExperimentalWarning
import warnings

# Filter out the specific experimental warning from Optuna
warnings.filterwarnings('ignore', category = ExperimentalWarning)

# This silences the trial-by-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)



# Start the clock!
start_time = time.perf_counter()

# Set the random state
random_state = 0 

# Load in data
temp1 = pd.read_csv(r'C:\Users\cramb\OneDrive\Desktop\Kaggle\Spaceship Titanic\train.csv')
temp2 = pd.read_csv(r'C:\Users\cramb\OneDrive\Desktop\Kaggle\Spaceship Titanic\test.csv')

# Concatenate data sets
titanic = pd.concat([temp1, temp2], ignore_index = True)

del temp1, temp2 

# Get the number missing
titanic['NumMissing'] = titanic.isna().sum(axis = 1)

# Spit Name
titanic[['FirstName', 'LastName']] = titanic['Name'].str.split(' ', expand = True)

# Split up Cabin
titanic[['Deck', 'CabinNum', 'Side']] = titanic['Cabin'].str.split('/', expand = True)

# Spit up passenger
titanic[['Group', 'Num']] = titanic['PassengerId'].str.split('_', expand = True)

# Get the group size
titanic['GroupSize'] = titanic.groupby('Group')['Group'].transform('count')

# Set passenter ID as the index
titanic = titanic.set_index('PassengerId', drop = True)

# Amenities per the kaggle page
amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Calculate the number of missing amenities, since these seem to have high NMI with transported
titanic['NumMissingAmenities'] = titanic[amenities].isna().sum(axis = 1)

# Calculate the median age
titanic['MedianAge'] = titanic.groupby('Group')['Age'].transform('median')

# Convert to numeric first (since it was split from a string)
titanic['CabinNum'] = pd.to_numeric(titanic['CabinNum'], errors = 'coerce')

# Convert to numeric first (since it was split from a string)
titanic['CabinBlock'] = titanic['CabinNum'] // 100

# Save starboard start
starboard_state = titanic['CabinBlock'].max() + 1

# The max cabin block is 18; add 19 if starboard side
titanic.loc[titanic['Side'] == 'S', 'CabinBlock'] += starboard_state

# Conver to string
titanic['CabinBlock'] = titanic['CabinBlock'].astype(str).replace('nan', np.nan)

# Drop columns we don't need
titanic = titanic.drop(columns = ['Cabin', 'CabinNum', 'Name'])

# Perform heuristic imputations of missing values
titanic = utils.apply_imputation_heuristics(titanic, amenities)

# Take the log of the amenities
titanic[amenities] = np.log1p(titanic[amenities])

# Create normalized mutual information data frame
nmi_df = utils.create_nmi_df(titanic, n_jobs = 18)  

print(titanic.isna().mean().sort_values())

feature_types = {
    # Categorical Features (Nominal or Binary)
    'HomePlanet': 'categorical',
    'CryoSleep': 'categorical',
    'Destination': 'categorical',
    'VIP': 'categorical',
    'Transported': 'categorical', # Target variable
    'FirstName': 'categorical',
    'LastName': 'categorical',
    'Deck': 'categorical',
    'Side': 'categorical',
    'Group': 'categorical', 
    'CabinBlock':'categorical',
    'Num': 'categorical', # The position within the group (e.g., 01, 02)

    # Numerical Features (Continuous or Discrete)
    'Age': 'numerical',
    'MedianAge':'numerical',
    'RoomService': 'numerical',
    'FoodCourt': 'numerical',
    'ShoppingMall': 'numerical',
    'Spa': 'numerical',
    'VRDeck': 'numerical',
    'NumMissing': 'numerical',
    'NumMissingAmenities':'numerical',
    'GroupSize': 'numerical'
}

# Search Spaces
search_params = {'pre__cat__encode__smoothing': FloatDistribution(1.0, 100.0, log = True),
                 'pre__cat__encode__min_samples_leaf': IntDistribution(1, 40)}

models_dict = {'reg':{
                        'xgb': (XGBRegressor(random_state = random_state, n_jobs = 1), {
                            'model__n_estimators': IntDistribution(50, 300),
                            'model__learning_rate': FloatDistribution(0.01, 0.2, log = True),
                            'model__max_depth': IntDistribution(3, 10),
                            **search_params
                            }),
                        'rf': (RandomForestRegressor(random_state = random_state, n_jobs = 1), {
                            'model__n_estimators': IntDistribution(50, 200),
                            'model__max_depth': IntDistribution(3, 20),
                            **search_params
                            }),
                        'en': (ElasticNet(max_iter = 5000), {
                            'model__alpha': FloatDistribution(1e-4, 100, log = True),
                            'model__l1_ratio': FloatDistribution(1e-5, 1 - 1e-5),
                            **search_params
                            })     
                        },
        'clf':{
                'xgb': (XGBClassifier(random_state = random_state, n_jobs = 1), {
                    'model__n_estimators': IntDistribution(50, 300),
                    'model__learning_rate': FloatDistribution(0.01, 0.2, log = True),
                    **search_params
                    }),
                'rf': (RandomForestClassifier(random_state = random_state, n_jobs=1), {
                    'model__n_estimators': IntDistribution(50, 200),
                    'model__max_depth': IntDistribution(3, 20),
                    **search_params
                    }),
                'log': (LogisticRegression(max_iter = 5000), {
                    'model__C': FloatDistribution(1e-4, 1e+2, log = True),
                    **search_params
                    })}
        }
        
    
# The Cascade Order
cascade_order = amenities + ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 
                             'Age', 'Deck']

imputers = {}

for target in cascade_order:
    
    start_time_inner = time.perf_counter()
    
    # Use 18 cores, but keep inner model n_jobs = 1 to avoid "Over-subscription"
    titanic, model, name = utils.run_systematic_imputation(titanic, 
                                                           target, 
                                                           nmi_df, 
                                                           feature_types, 
                                                           models_dict,
                                                           n_jobs = 18, 
                                                           random_state = 0)
    
    imputers[target] = (name, model)
    
    # Update nmi_df
    nmi_df = utils.create_nmi_df(titanic, n_jobs = 18)
    
    gc.collect()
    
    print(f'Imputation for {target} is done via {name}. It took {(time.perf_counter() - start_time_inner)/60:0.2f} minutes.')

# Delete random forest to save time (ran before)
del models_dict['clf']['rf']

# Imput the block
titanic = utils.bayesian_block_imputer(titanic, nmi_df, feature_types, 
                                       models_dict, starboard_state,
                                        nmi_threshold = 0.02, n_jobs = 18, 
                                        random_state = random_state)

# Make CabinBlock numeric for the main calculation
titanic['CabinBlock'] = titanic['CabinBlock'].astype(float)

# Calculate the side
titanic['Side'] = np.where(titanic['CabinBlock'] >= starboard_state, 'S', 'P')
    
# Calculate the sum of the amenities
titanic['Amenities'] = np.log1p(np.expm1(titanic[amenities]).sum(axis = 1))

# Plot the fraction transported by deck and cabin block
utils.plot_ship_transport_heatmap(titanic, starboard_state)

# Based on image make variable to describe anomaly; about 71% of people in an anomaly are transported

# The Bow Hit (Port and Starboard Front, Upper Decks)
titanic['BowAnomaly'] = (
                        (titanic['Deck'] == 'B') | 
                        ((titanic['Deck'] == 'C') & ((titanic['CabinBlock'] < 3)|(titanic['CabinBlock'] >= 19)))
                        ).astype(int)
    
# The Starboard Scrape (Mid-Front, Lower Decks)
titanic['StarboardScrape'] = (
                            ((titanic['Deck'] == 'E') & (titanic['CabinBlock'] == 25))|
                            ((titanic['Deck'] == 'F') & titanic['CabinBlock'].between(25, 29))|
                            ((titanic['Deck'] == 'G') & titanic['CabinBlock'].between(25, 30))
                            ).astype(int)

# Create a feature that interacts between amenities and the anomalies
titanic['LuxuryTrap'] = (titanic['BowAnomaly']|titanic['StarboardScrape']
                                ) & (titanic['Amenities'] > titanic['Amenities'].median())

# Create normalized mutual information data frame
nmi_df = utils.create_nmi_df(titanic)  

plt.figure(figsize = (14, 12)) 

sns.heatmap(nmi_df, annot = True, cmap = 'coolwarm', fmt = '0.2f')    

plt.title('Normalized Mutual Information Final')

plt.show()

# Define the variables
num_vars = ['Age', 'Amenities', 'CabinBlock', 'FoodCourt', 'GroupSize', 
            'MedianAge', 'NumMissing', 'NumMissingAmenities', 'RoomService', 
            'ShoppingMall',  'Spa', 'VRDeck']
cat_vars = ['Deck', 'Destination', 'HomePlanet']
ohe_vars = ['BowAnomaly', 'CryoSleep', 'LuxuryTrap', 'Side', 'StarboardScrape', 
            'VIP']

# Get all the variables
all_vars = num_vars + cat_vars + ohe_vars

# Make sure categorical features are strings
for col in cat_vars + ohe_vars:
    
    titanic[col] = titanic[col].astype(str)
    
# Breack up the titanic data frame
titanic_train = titanic.loc[titanic['Transported'].notna(), :]
titanic_test = titanic.loc[titanic['Transported'].isna(), :]  

del titanic

# Cast the target to integer 
titanic_train['Transported'] = titanic_train['Transported'].astype(int)

# Define the search space for our base estimators
search_spaces = {
            'cat__depth': IntDistribution(4, 10),
            'cat__learning_rate': FloatDistribution(0.01, 0.2, log = True),
            'cat__iterations': IntDistribution(100, 500),
            'cat__l2_leaf_reg': FloatDistribution(0.01, 25, log = True), 
            'cat__random_strength': FloatDistribution(1e-9, 20, log = True),
            'cat__subsample': FloatDistribution(0.5, 1.0),  
            'cat__colsample_bylevel': FloatDistribution(0.5, 1.0), 
            'cat__bootstrap_type': CategoricalDistribution(['Bernoulli', 'MVS'])
            }

# Start the block for cat boost!
cat_start = time.perf_counter()

# CatBoost Gatekeeper (Raw Values)
cat_pre = ColumnTransformer(transformers = [
        ('keep', 'passthrough', num_vars + cat_vars + ohe_vars)], 
    remainder = 'drop')

# Specify the categorical features for catboost
cat_indices = list(range(len(num_vars), len(all_vars)))

# Define statified cv
strat_cv = StratifiedKFold(n_splits = 5, 
                           shuffle = True, 
                           random_state = random_state)

# Define the Base Models
model = CatBoostClassifier(verbose = 0, 
                           allow_writing_files = False, 
                           cat_features = cat_indices, 
                           random_state = random_state)
    
# Construct the pipeline
pipe = Pipeline([('pre', cat_pre), ('cat', model)])
            
# Bayesian Optimization
opt_stack = OptunaSearchCV(estimator = pipe,
                           param_distributions = search_spaces,
                           n_trials = 60,
                           cv = strat_cv,
                           scoring = 'accuracy',
                           error_score = 'raise',
                           random_state = random_state,
                           n_jobs = 18)

opt_stack.fit(titanic_train[all_vars], titanic_train['Transported'])

print(f'It took catboost {(time.perf_counter() - cat_start)/60 :.2f} minutes.')

# Extract the best CatBoost model from the pipeline
best_cat_model = opt_stack.best_estimator_.named_steps['cat']

# Extract the feature names from the preprocessor
feature_names = opt_stack.best_estimator_.named_steps['pre'].get_feature_names_out()

# Get the importance values
importances = best_cat_model.get_feature_importance()

# Create a readable Series
feat_imp = pd.Series(importances, index = feature_names).sort_values(ascending = False)

print("Top 10 Features (CatBoost):")
print(feat_imp.head(10))

# Delete these so I don't get confused
del best_cat_model, feature_names, importances 

# Perform audit
audit_df = utils.audit_results(titanic_train[all_vars], 
                               titanic_train['Transported'], 
                               opt_stack.best_estimator_, 
                               strat_cv)

# Print the audit data frame
audit_df.to_csv('audit.csv', index = False)

# Plot to see whether we are separating classes well
utils.plot_confidence_audit(audit_df)

# Plot accuracy by feature
utils.plot_feature_bias_audit(audit_df, titanic_train, feature = 'Amenities')
utils.plot_feature_bias_audit(audit_df, titanic_train, feature = 'BowAnomaly')
utils.plot_feature_bias_audit(audit_df, titanic_train, 
                              feature = 'StarboardScrape')

# Get the probabilities
pred_probas = opt_stack.predict_proba(titanic_test[all_vars])[:, 1]

# Get the final predictions
final_preds = utils.apply_spaceship_group_mask(titanic_train, titanic_test, 
                                               pred_probas, upper = 0.75, 
                                               lower = 0.25)

# Check portion that servived
print(final_preds.value_counts(normalize = True))

# Create the submission data frame
submission_df = pd.DataFrame({
                            'PassengerId': titanic_test.index,
                            'Transported': final_preds.astype(bool) 
                            })

# Save to CSV
submission_df.to_csv('submission_v2.csv', index = False)

print(f'This program took {(time.perf_counter() - start_time)/60:0.2f} minutes.')
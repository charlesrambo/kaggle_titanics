# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 17:51:08 2026

@author: cramb
"""
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.compose import ColumnTransformer
import seaborn as sns
import time

# Utility script
from titanic_utils import engineer_features, AgeImputer, get_nmi, apply_group_mask

# Load the data into notebook
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

# Set the random state
random_state = 0

# Create marker to denote whether it's test data
titanic_train['IsTest'] = False
titanic_test['IsTest'] = True

# Combine files
titanic_full = pd.concat([titanic_train, titanic_test], axis = 0, ignore_index = True)

# Check out the columns and how many are missing
for col in titanic_full:

    print(f'{col} has {titanic_full[col].isna().sum()} missing values.')
    
# Create new features
titanic_full = engineer_features(titanic_full)

# Check out the columns and how many are missing
for col in titanic_full:

    print(f'{col} has {titanic_full[col].isna().sum()} missing values.')
    
# See what features have the highest mutual information with age
# See what features have the highest mutual information with age
get_nmi(titanic_full, 
        x_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 
                     'PartyNumber', 'FamilySize', 'IndividualFare', 'TicketPrefix', 
                     'Deck', 'ExitDistance', 'CabinCount', 'Starboard'], 
        y_col = 'Age')

# Identify feature types
num_features = ['IndividualFare', 'Parch', 'SibSp', 'Pclass', 
                'PartyNumber', 'FamilySize', 'CabinCount']
cat_features = ['Title', 'Deck']


# Create Preprocessing Layers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('target', ce.TargetEncoder(), cat_features)
    ])

# Define the search spaces with explicit skopt types
# Note the 'model__' prefix to reach through the pipeline to the regressor
regressors = {
    'XGBoost': (XGBRegressor(random_state = random_state), {
        'model__n_estimators': Integer(50, 300),
        'model__max_depth': Integer(3, 10),
         # L1 regularization
        'model__reg_alpha': Real(1e-5, 10, prior='log-uniform'),
        # L2 regularization
        'model__reg_lambda': Real(1e-5, 10, prior='log-uniform'), 
        'model__learning_rate': Real(0.01, 0.2, prior = 'log-uniform'),
        'model__subsample': Real(0.5, 1.0),
        'model__colsample_bytree': Real(0.5, 1.0),
        'pre__target__smoothing':Integer(5, 20)
    }), 
    'RandomForest': (RandomForestRegressor(random_state = random_state), {
        'model__max_features': Categorical(['sqrt', 1.0]),
        'model__ccp_alpha': Real(1e-5, 1e-2, prior = 'log-uniform'),
        'model__n_estimators': Integer(10, 200),
        'model__max_depth': Integer(3, 20),
        'pre__target__smoothing':Integer(5, 20)
    }),
    'ElasticNet': (ElasticNet(), {
        'model__alpha': Real(1e-4, 100, prior = 'log-uniform'),
        'model__l1_ratio': Real(0, 1),
        'pre__target__smoothing':Integer(5, 20)
    }),
    'SVM': (SVR(), {
    'model__kernel': Categorical(['linear', 'rbf']),
    'model__C': Real(0.1, 100, prior='log-uniform'),
    'model__gamma': Real(1e-3, 0.1, prior='log-uniform'),
    'model__epsilon': Real(0.01, 0.5),
    'pre__target__smoothing':Integer(5, 20)
    })
}

# Get the training data
X_train = titanic_full.loc[titanic_full['Age'].notna(), num_features + cat_features]
y_train = titanic_full.loc[titanic_full['Age'].notna(), 'Age']

# Execution Loop
results = {}

for name, (model, params) in regressors.items():
    
    print(f"Optimizing {name}...")

    # Construct the pipeline; SVM needs results to be at about same scale, so adding final_scale
    pipe = Pipeline([('pre', preprocessor), 
                     ('final_scale', StandardScaler()), 
                     ('model', model)])

    # Initialize optimization object
    opt = BayesSearchCV(pipe, params, n_iter = 32, cv = 5, 
                        scoring = 'r2', n_jobs = -1, 
                        random_state = random_state)

    # Fit the model
    opt.fit(X_train, y_train)

    # Record the results
    results[name] = {
        'best_score': opt.best_score_,
        'best_model': opt.best_estimator_
    }

print(r"'Optimizing' Mean...")

baseline_pipe = Pipeline([('pre', preprocessor), ('model', DummyRegressor(strategy = 'mean'))])

# Mean Imputer as baseline
mean_score = cross_val_score(baseline_pipe, X_train, y_train, 
                             cv = KFold(n_splits = 5, shuffle = True, random_state = random_state), scoring = 'r2').mean()

# Record results
results['Mean'] = {'best_score': mean_score, 'best_model': 'Simple Mean'}

del X_train, y_train

print("\n--- Final R^2 Scores ---")

# Output Results Comparison
for name, res in results.items():
    
    print(f"{name} R^2 Score: {res['best_score']:.4f}")
    
# Find the key with the highest best_score
winner_name = max(results, key = lambda x: results[x]['best_score'])
winner_pipeline = results[winner_name]['best_model']

print(f'Applying predictions from the winner: {winner_name}  \n')

imputer = AgeImputer(winner_pipeline, cat_features, num_features)

titanic_full = imputer.transform(titanic_full)

# Get feature importances if they exist
if hasattr(winner_pipeline.named_steps['model'], 'feature_importances_'):

    print('In-sample feature importances:')

    # Re-construct the feature names list 
    
    # Get names from the OneHotEncoder
    cat_encoder = winner_pipeline.named_steps['pre'].transformers_[1][1]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    
    # Final list: numeric features first, then the new encoded categorical names
    full_feature_names = num_features + encoded_cat_names
    
    # Zip with the actual expanded feature list
    importances = winner_pipeline.named_steps['model'].feature_importances_

    # Loop over results
    for col, imp in zip(num_features + cat_features, importances):

        print(f'{col}: {imp: .4f}')
        
    del importances, cat_encoder, encoded_cat_names, full_feature_names


# Check out the columns and how many are missing; everything should be filled!
for col in titanic_full:

    print(f'{col} has {titanic_full[col].isna().sum()} missing values.')
    
# Break up data again
titanic_train = titanic_full.loc[~titanic_full['IsTest'], :].drop(columns = ['IsTest'])
titanic_test = titanic_full.loc[titanic_full['IsTest'], :].drop(columns = ['IsTest'])

# Delete data frame that has done its job
del titanic_full

# Take a look at normalized mutual information
print(get_nmi(titanic_train, 
        x_columns = ['Age', 'Pclass', 'Sex', 'SibSp',
                        'Parch', 'Embarked', 'Title', 
                         'PartyNumber', 'FamilySize', 'IndividualFare', 
                         'TicketPrefix', 'Deck', 'NumMissing',
                        'ExitDistance', 'CabinCount', 'Starboard'], 
        y_col = 'Survived', random_state = random_state))

#  Survival as a function of Age, faceted by Title
titles = titanic_train['Title'].unique()

fig, axes = plt.subplots(1, len(titles), figsize = (20, 5), sharey  =True)

for i, title in enumerate(titles):

    # Subset data
    subset = titanic_train[titanic_train['Title'] == title]
    
    # The 'logistic=True' creates that classic sigmoid probability curve
    sns.regplot(x = 'Age', y = 'Survived', data = subset, 
                logistic = True, ax = axes[i], 
                scatter_kws = {'alpha':0.3, 'color':'gray'},
                line_kws = {'color':'firebrick'})
    
    axes[i].set_title(f'Survival Prob: {title}')
    axes[i].set_ylim(-0.05, 1.05)
    axes[i].grid(axis = 'y', linestyle = '--', alpha = 0.7)

plt.tight_layout()

del subset 

plt.show()


# Survival as a function of IndividualFare
plt.figure(figsize = (10, 6))

sns.regplot(x = 'IndividualFare', y = 'Survived', data = titanic_train, 
            logistic = True,
            scatter_kws = {'alpha':0.2},
            line_kws = {'color':'navy'})

plt.title('Survival Probability vs. Individual Fare')

plt.xlabel('Individual Fare (Un-pooled)')

plt.ylabel('Probability of Survival')

plt.xscale('log') # Fares often look better on a log scale

plt.grid(True, which = "both", ls="-", alpha = 0.5)

plt.show()

# Standard Models Gatekeeper (Handles the Encoding)
standard_pre = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Age', 'SibSp', 'Parch', 'IndividualFare', 'PartyNumber', 'FamilySize', 'NumMissing', 'CabinCount', 'ExitDistance']),
        ('ohe', OneHotEncoder(drop = 'first'), ['Sex', 'Pclass', 'Embarked', 'Starboard']),
        ('target', ce.TargetEncoder(), ['Title', 'Deck', 'TicketPrefix'])
    ],
    remainder = 'drop'
)

# CatBoost Gatekeeper (Raw Values)
cat_pre = ColumnTransformer(
    transformers=[
        ('keep', 'passthrough', ['Age', 'SibSp', 'Parch', 'IndividualFare', 'PartyNumber', 'FamilySize', 'NumMissing', 'CabinCount', 'ExitDistance',
                                'Sex', 'Pclass', 'Embarked', 'Starboard',
                                'Title', 'Deck', 'TicketPrefix'])
    ],
    remainder='drop'
)

# Define the columns for each group
num_features = ['Age', 'SibSp', 'Parch', 'IndividualFare', 'PartyNumber', 'FamilySize', 'NumMissing', 'CabinCount', 'ExitDistance']
cat_features = ['Sex', 'Pclass', 'Embarked', 'Starboard', 'Title', 'Deck', 'TicketPrefix']

# Construct the Master X_train
# Ensure categorical columns are strings for the pipelines
all_features = num_features + cat_features
X_train = titanic_train[all_features].copy()

# Final type check to ensure no surprises during fit
for col in cat_features:
    
    X_train[col] = X_train[col].astype(str)

y_train = titanic_train['Survived']

# Specify the categorical features for catboost
cat_indices = list(range(len(num_features), 
                         len(num_features) + len(cat_features)))

# Define the search space for our base estimators
search_spaces = {
    'xgb': {
        'model__n_estimators': Integer(50, 500),
        'model__max_depth': Integer(3, 10),
        'model__learning_rate': Real(0.01, 0.3, prior = 'log-uniform'),
        # L1 and L2 regularization for boosting
        'model__reg_alpha': Real(1e-5, 10, prior = 'log-uniform'),
        'model__reg_lambda': Real(1e-5, 10, prior = 'log-uniform'),
        'model__subsample': Real(0.5, 1.0, prior = 'uniform'),
        'model__colsample_bytree': Real(0.5, 1.0, prior = 'uniform'),
        'pre__target__smoothing': Integer(5, 20)
    },
    'cat': {
        'model__depth': Integer(4, 10),
        'model__learning_rate': Real(0.01, 0.2, prior = 'uniform'),
        'model__iterations': Integer(100, 500),
        'model__l2_leaf_reg': Real(0.01, 20, prior = 'log-uniform'), 
        'model__random_strength': Real(1e-9, 20, prior = 'log-uniform'),
        'model__subsample': Real(0.5, 1.0, prior = 'uniform'),  
        'model__colsample_bylevel': Real(0.5, 1.0, prior = 'uniform'), 
        'model__bootstrap_type': Categorical(['Bernoulli', 'MVS'])
    },
    'rf': {
        'model__n_estimators': Integer(100, 500),
        'model__max_depth': Integer(5, 20),
        'model__max_features': Categorical(['sqrt', 'log2']),
        'model__min_samples_split': Integer(2, 10),
        'model__ccp_alpha': Real(1e-5, 1e-2, prior = 'log-uniform'),
        'pre__target__smoothing': Integer(5, 20)
    },
    'svc': {
        'model__C': Real(0.1, 50, prior = 'log-uniform'),
        'model__gamma': Real(1e-4, 0.1, prior = 'log-uniform'),
        'model__kernel': Categorical(['rbf', 'linear']),
        'pre__target__smoothing': Integer(5, 20),
        'selector__n_features_to_select': Integer(4, len(all_features) - 6),
    },
    'log': {
        'model__C': Real(1e-4, 1e+2, prior = 'log-uniform'),
        'model__penalty': Categorical(['l2']),
        'model__solver': Categorical(['lbfgs', 'newton-cg']),
        'pre__target__smoothing': Integer(5, 20)
    }
}

# Define the Base Models
base_models_to_tune = [
    ('xgb', XGBClassifier(eval_metric = 'logloss', random_state = random_state)),
    ('cat', CatBoostClassifier(verbose = 0, allow_writing_files = False, 
                               cat_features = cat_indices, random_state = random_state)),
    ('rf', RandomForestClassifier(random_state = random_state)),
    ('svc', SVC(probability = True, random_state = random_state)),
    ('log', LogisticRegression(max_iter = 1000))
]

# Initialize list to hold the runed estimators
tuned_base_estimators = []

# Initialize dictionary to hold predictions
base_model_predictions = {}

# The Training Loop: Individual Optimization
print("Starting individual model optimization...")

for name, model in base_models_to_tune:

    # Start the clock!
    start_time = time.perf_counter()

    print(f"Optimizing {name}...")

    # Define number of jobs
    n_jobs = 18 if name == 'cat' else -1

    # Define the selector
    if name not in ['svc', 'xgb']:
        
        selector = RFECV(estimator = model, 
                         step = 1,
                         cv =  StratifiedKFold(n_splits = 10, shuffle = True, random_state = random_state), 
                         scoring = 'accuracy', n_jobs = n_jobs)
        
        n_iter = 40
        
    elif name == 'xgb':
        
        # Oddly, the extra computations didn't help xgboost
        selector = RFECV(estimator = model, 
                         step = 2,
                         cv =  StratifiedKFold(n_splits = 10, shuffle = True, random_state = random_state), 
                         scoring = 'accuracy', n_jobs = n_jobs)        
        
        n_iter = 20
        
    else:
        
        n_iter = 40

        # SVC doesn't support REFCV
        selector = SequentialFeatureSelector(estimator = model, 
                                             cv = StratifiedKFold(n_splits = 5, shuffle = True, 
                                                                  random_state = random_state), 
                                             scoring = 'accuracy',
                                             tol = 0.001,
                                             n_jobs = n_jobs)

    # Choose the correct gatekeeper
    if name == 'cat':

        # Construct the pipeline
        pipe = Pipeline([('pre', cat_pre), 
                         ('model', model)])
        
        # Catboost isn't doing feature selection so increase
        n_iter = 50

    else:


        # Construct the pipeline
        pipe = Pipeline([('pre', standard_pre), 
                        ('final_scale', StandardScaler()), 
                         ('selector', selector),
                         ('model', model)])  
        
        
    # We wrap in a simple dict-to-BSCV logic
    opt = BayesSearchCV(
        pipe,
        search_spaces.get(name, {}), 
        n_iter = n_iter, 
        cv = 5, 
        n_jobs = n_jobs,
        scoring = 'neg_log_loss',
        random_state = random_state
    )
    
    # Assuming X_train, y_train are ready
    opt.fit(X_train, y_train)

    # Add results to list
    tuned_base_estimators.append((name, opt.best_estimator_))

    # Get cross-validation probabilities
    probs = cross_val_predict(opt.best_estimator_, X_train, y_train, 
                              cv = KFold(n_splits = 10, shuffle = True, random_state = random_state), 
                              method = 'predict_proba')[:, 1]

    # Save the probabilities
    base_model_predictions[name] = probs

    # Calculate Performance Metrics
    ce_score = log_loss(y_train, probs)
    acc_score = accuracy_score(y_train, (probs > 0.5).astype(int))
    
    print(f"{name} Results -> Accuracy: {acc_score:.4f}, Cross-Entropy: {ce_score:.4f}")

    # Print time
    print(f'It took {name} a total of {(time.perf_counter() - start_time)/60:.2f} minutes.')

# Convert to pandas data frame
base_model_predictions = pd.DataFrame.from_dict(base_model_predictions)

# Diversity Check: Correlation Heatmap
plt.figure(figsize = (10, 8))

sns.heatmap(base_model_predictions.corr(), annot = True, cmap = 'coolwarm', fmt = ".2f")

plt.title("Base Model Prediction Correlations")

plt.show()

# SAVE RESULTS!
import joblib

print("\n" + "="*50)
print("KAGGLE COPY-PASTE READY RESULTS")
print("="*50 + "\n")

# Store results in a dictionary for easy export
final_export = {}

for name, model in tuned_base_estimators:
    
    # Extract Selected Features
    if 'selector' in model.named_steps:
        
        selector = model.named_steps['selector']
        
        # Handle different selector types
        if hasattr(selector, 'support_'):
            
            # RFECV uses .support_
            support = selector.support_.tolist()
            
            
            
        else:
            
            # SFS uses .get_support()
            support = selector.get_support()
          
        # Get names for the selected features
        print(f"{name}_support_mask = {support}")
        
    else:
        # CatBoost uses all features        
        support = [True] * len(all_features)
    
    # Extract Hyperparameters
    params = model.named_steps['model'].get_params()
    
    # Optional: Filter for only the most important params to keep Kaggle cell clean
    # (Comment this out if you want every single default parameter)
    keys_to_keep = ['n_estimators', 'learning_rate', 'max_depth', 'C', 'gamma', 'kernel', 'l2_leaf_reg', 'iterations']
    filtered_params = {k: v for k, v in params.items() if k in keys_to_keep or 'model__' in k}

    final_export[name] = {
        'features': support,
        'params': params
    }
    
    print(f"# {name.upper()} CONFIG")
    print(f"{name}_best_features = {support}")
    print(f"{name}_best_params = {params}\n")
    print("-" * 30)

joblib.dump(tuned_base_estimators, 'final_tuned_models2.pkl')

# Start the clock!
start_time = time.perf_counter()

# Filter our tuned estimators to the 'Lean Three'
final_base_models = [
    (name, estimator) for name, estimator in tuned_base_estimators 
    if name in ['xgb', 'cat', 'rf', 'svc']
]

# Define statified cv
strat_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)

# Define the Stacking Classifier
stacking_clf = StackingClassifier(
    estimators = final_base_models,
    final_estimator = LogisticRegression(solver = 'lbfgs', max_iter = 1000),
    stack_method = 'predict_proba', # or your LogitTransformer pipeline later
    cv = strat_cv,
    n_jobs = 1
)

# Define the Bayesian search space
# Using a log-uniform prior for C is much more efficient than a grid
stack_params = {
    'final_estimator__C': Real(1e-3, 1e2, prior = 'log-uniform'),
    'final_estimator__fit_intercept': Categorical([True, False])
}

# Bayesian Optimization
opt_stack = BayesSearchCV(
    stacking_clf, 
    stack_params, 
    n_iter = 40, 
    cv = 5, 
    scoring = 'accuracy',
    random_state = random_state,
    n_jobs = -1
)

opt_stack.fit(X_train, y_train)

# Print time
print(f'It took the stacker a total of {(time.perf_counter() - start_time)/60:.2f} minutes.')

import copy

estimator_copy = copy.deepcopy(opt_stack.best_estimator_)

# Generate Out-Of-Fold predictions for the training set
# This gives us a 'Survival Probability' for every person in the training data
oof_probs = cross_val_predict(estimator_copy, X_train, y_train, cv = strat_cv, method = 'predict_proba')[:, 1]

#  Attach probabilities to the training dataframe for analysis
audit_df = X_train.copy()
audit_df['Actual'] = y_train
audit_df['Prob_Survived'] = oof_probs
audit_df['Error'] = abs(audit_df['Actual'] - audit_df['Prob_Survived'])
audit_df = audit_df.sort_values(by = 'Error', ascending = False)

audit_df.to_csv('audit.csv', index = False)

# Look at the "Biggest Lies" (High Error)
# These are the passengers the model is most wrong about
biggest_lies = audit_df.head(10)

print("Top 10 Passengers the Stacked Model Failed On:")
print(biggest_lies[['Title', 'Pclass', 'Sex', 'IndividualFare', 'Actual', 'Prob_Survived']])

# Access the fitted final estimator using dot notation
# Note: 'final_estimator_' is the fitted version of the meta-learner
meta_model = opt_stack.best_estimator_.final_estimator_

# Get the coefficients (weights)
raw_weights = meta_model.coef_[0]

# Normalize to see relative contribution (0% to 100%)
importance = np.abs(raw_weights) / np.sum(np.abs(raw_weights))

# Map them to the names of your "Lean Three"
model_names = [name for name, _ in final_base_models]
weight_dict = dict(zip(model_names, importance))

print("Meta-Learner Relative Contribution:")
for name, weight in weight_dict.items():
    print(f"{name}: {weight:.2%}")
    
# Get X_test
X_test = titanic_test[all_features].copy()

# Final type check to ensure no surprises during fit
for col in cat_features:
    
    X_test[col] = X_test[col].astype(str)

# Get whether survived
pred_proba = opt_stack.predict_proba(X_test)[:, 1]

# Apply the group mask
titanic_test['Survived'] = apply_group_mask(titanic_train, titanic_test, pred_proba)

# Check portion that servived
print(titanic_test['Survived'].value_counts(normalize = True))

# Create the submission data frame
submission_df = titanic_test[['PassengerId', 'Survived']]

# Cast Survived to int (Kaggle expects 0 or 1, sometimes model.predict returns floats)
submission_df['Survived'] = submission_df["Survived"].astype(int)

# Save to CSV
submission_df.to_csv('submission.csv', index = False)

print("Submission file saved successfully!")
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 17:58:26 2026

@author: charles rambo
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import entropy


def get_distance_from_stairs(num, deck):
    
    # Handle missing cabin numbers or 'U' labels
    if num == -1 or pd.isna(num):
        
        # Penalty value for unknown location
        return 100  
    
    # Anchor Points (Historical Cabin Numbers closest to Stairs/Elevators)
    # Forward Grand Staircase & Elevators are roughly at #55
    # Aft Grand Staircase is roughly at #100-110
    
    if deck == 'A':
        # On A-Deck, the stairs were at the very front and mid-aft.
        # Cabin A-37 was right at the stairs.
        return np.abs(num - 37)
    
    elif deck in ['B', 'C']:
        # B and C decks had cabins wrapped around the Forward (55) and Aft (100) stairs.
        dist_fwd = np.abs(num - 55)
        dist_aft = np.abs(num - 105)
        return np.min([dist_fwd, dist_aft])
    
    elif deck == 'D':
        # D-Deck Forward stairs were at #20. Aft stairs near #100.
        dist_fwd = np.abs(num - 20)
        dist_aft = np.abs(num - 100)
        return np.min([dist_fwd, dist_aft])
    
    elif deck == 'E':
        # E-Deck was a maze. Scotland Road (crew/steerage) was here.
        # Main stairs roughly at #50 and #110.
        dist_fwd = np.abs(num - 50)
        dist_aft = np.abs(num - 110)
        return np.min([dist_fwd, dist_aft])
    
    elif deck in ['F', 'G']:
        # Deep decks. Stair access was at the extreme ends (Bow/Stern).
        # We'll use 10 and 150 as proxies for the "ends" of the habitable areas.
        dist_bow = np.abs(num - 10)
        dist_stern = np.abs(num - 150)
        return np.min([dist_bow, dist_stern])
    
    # Default for decks like 'T' or 'Unknown'
    return 80 


def get_cabin_count(cabin_str):
    
    # If missing, probably lower class so return 0
    if pd.isna(cabin_str):
        
        return 0
    
    # The cabins are seperated by spaces
    else:
        
        return len(str(cabin_str).split())

def engineer_features(df):
    """
    Apply global feature engineering. 
    NOTE: Should be run on the combined (train+test) dataframe 
    to get accurate PartyNumber and IndividualFare.
    """
    
    # Create dictionary to convert titles
    title_dict = {'Don':'Mr', 'Dona':'Mrs', 'Mme':'Mrs', 'Ms':'Miss', 
                  'Major':'Mr', 'Lady':'Mrs', 'Sir':'Mr', 'Mlle':'Miss', 
                  'Col':'Mr', 'Capt':'Mr', 'the Countess':'Mrs', 'Jonkheer':'Mr'}

    # Create a dictionary to simplify the ticket prefix
    prefix_dict = {
        # The 'SOTON' / 'STON' Family (Southampton variants)
        'SOTONO': 'SOTON', 
        'SOTONOQ': 'SOTON', 
        'STONO': 'SOTON',
        
        # The 'SC' / 'Paris' Family (Société Centrale / Paris variants)
        'SCPARIS': 'SC_PARIS',
        'SCParis': 'SC_PARIS',
        'SCAH Basle': 'SCAH', 
        'SCA': 'SCAH',
        'SOPP': 'SOP',
        
        # The 'FC' Family (Likely 'First Class' variations)
        'FCC': 'FC',
        
        # The 'PP' Family (Likely 'Passenger Priority' or 'Pre-Paid')
        'PPP': 'PP',
        
        # Group these into 'Other'
        'AS': 'Other', 'Fa': 'Other', 'SP': 'Other', 'SCOW': 'Other'
    }

    
    # Create column to denote number of missing vars for observation
    df['NumMissing'] = df.isna().sum(axis = 1)
    
    # Convert Sex to 1 or 0
    df['Sex'] = (df['Sex'] == 'female').astype(int)

    # Get the title; dataset has a very consistent struction
    df['Title'] = df['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
    
    # Convert male doctors to Mr and female doctors to Mrs, because ML model doesn't do well on these
    df.loc[(df['Title'] == 'Dr') & (df['Sex'] == 'male'), 'Title'] = 'Mr'
    df.loc[(df['Title'] == 'Dr') & (df['Sex'] == 'female'), 'Title'] = 'Mrs'

    # Convert titles so we don't overfit
    df['Title'] = df['Title'].replace(title_dict)
    
    # Get the family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Get the count for each ticket; this is the group size
    df['PartyNumber'] = df.groupby('Ticket')['Ticket'].transform('count')
    
    # Get the Individual Fare
    df['IndividualFare'] = df['Fare'] / df['PartyNumber']
    
    # Extract non-digits from the beginning of the string
    df['TicketPrefix'] = df['Ticket'].str.extract(r'^(\D+)', expand = False)

    # Clean up: remove dots/slashes and extra whitespace
    df['TicketPrefix'] = df['TicketPrefix'].str.replace(r'[\./]', '', regex = True).str.strip()

    # Fill missing values (for tickets that were just numbers) with 'Numeric'
    df['TicketPrefix'] = df['TicketPrefix'].fillna('Numeric')

    # Use dictionary to convert prefix
    df['TicketPrefix'] = df['TicketPrefix'].replace(prefix_dict)

    # Get the counts
    df['count'] = df.groupby('TicketPrefix')['TicketPrefix'].transform('count')
    
    # Replace all rare ones with a single label
    df.loc[df['count'] < 5, 'TicketPrefix'] = 'Rare'

    # Impute based on TicketPrefix
    prefix_mode = df.groupby('TicketPrefix')['Embarked'].apply(lambda x: x.mode()[0])

    # Fill the two missing values with the mode
    df['Embarked'] = df['Embarked'].fillna(df['TicketPrefix'].map(prefix_mode))
    
    # Identify the missing fare observation
    missing_fare = df['IndividualFare'].isna()
    
    if missing_fare.any():
        
        # Calculate the mean fare based on Parch and TicketPrefix
        df['IndividualFare'] = df['IndividualFare'].fillna(
            df.groupby(['Pclass', 'TicketPrefix'])['IndividualFare'].transform('mean'))
        
        # If that specific combination is also empty use the Pclass mean
        if df['IndividualFare'].isna().any():
            df['IndividualFare'] = df['IndividualFare'].fillna(df.groupby('Pclass')['IndividualFare'].transform('mean'))
    
    # Take log to make it more normal
    df['IndividualFare'] = np.log1p(df['IndividualFare'])
    
    # Create the initial Deck column
    df['Deck'] = df['Cabin'].str[0]

    # People on the same ticket usually share a deck
    ticket_first = df.groupby('Ticket')['Deck'].first() 
    df['Deck'] = df['Deck'].fillna(df['Ticket'].map(ticket_first))

    # Impute based on Pclass (General Accuracy)
    pclass_mode = df.groupby('Pclass')['Deck'].apply(lambda x: x.mode()[0])
    df['Deck'] = df['Deck'].fillna(df['Pclass'].map(pclass_mode))

    # Final safety net (not used!)
    df['Deck'] = df['Deck'].fillna('U')
    
    # Extract the numeric part of the cabin
    df['CabinNumber'] = df['Cabin'].str.extract('(\d+)').astype(float)
    
    # Get the corresponding distance
    df['ExitDistance'] = [get_distance_from_stairs(num, deck) for num, deck in zip(df['CabinNumber'], df['Deck'])]
    
    # Get the cabin count
    df['CabinCount'] = df['Cabin'].apply(get_cabin_count)
    
    # Get whether it's 'Starboard'
    df['Starboard'] = (df['CabinNumber'] % 2).fillna(-1).astype(int)
    
    # Drop column that has done its job
    df = df.drop(columns = ['count', 'Fare', 'Cabin', 'CabinNumber'])

    return df


# Create function to get mutual information
def get_nmi(df, x_columns, y_col, random_state = 0):

    # Get the columns for analysis
    df_mi = df[x_columns + [y_col]].copy()
    
    # We use .factorize() to turn strings into numbers
    for col in df_mi.select_dtypes("object"):
        
        df_mi[col], _ = df_mi[col].factorize()
    
    # Fill missing values with -1
    df_mi = df_mi.fillna(-1)

    # Separate target from your features
    X = df_mi.drop(y_col, axis = 1)
    y = df_mi[y_col]
    
    # Identify which columns are "discrete" (the ones we factorized)
    discrete_features = X.dtypes == int

    # Logic to choose Classification vs Regression
    # If y has few unique values or is an object/bool, use classif
    if y.dtype == 'object' or y.nunique() < 10:
        
        mi_func = mutual_info_classif
        
    else:
        
        mi_func = mutual_info_regression
    
    #  Calculate MI Scores
    mi_scores = mi_func(X, y, discrete_features = discrete_features, 
                                       random_state = random_state)

    # Entropy of the target
    target_entropy = entropy(y.value_counts(normalize=True))
    
    # Organize the results
    nmi_results = pd.Series(mi_scores/target_entropy, name = "MI Scores", index = X.columns)

    # Sort values from largest to smallest
    nmi_results = nmi_results.sort_values(ascending = False)
    
    return nmi_results


class AgeImputer:
    """
    A domain-aware Age Imputer that combines Gradient Boosting 
    with 1912 historical social norms.
    """
    def __init__(self, model, cat_features, num_features, fix_preds = True):
        
        # Save the model
        self.model = model
        
        # The specific columns you defined
        self.cat_features = cat_features
        self.num_features = num_features
        self.all_features = num_features + cat_features
        
        # Will we write over predictions?
        self.fix_preds = fix_preds

    def fit(self, X, y = None):
        
        # This is just for scikit-learn compatibility
        return self

    def transform(self, X):
        
        # Create a copy so we don't accidentally overwrite our original df
        X = X.copy()
        
        # Get the missing ages
        missing_age = X['Age'].isna()
        
        if missing_age.any():
            
            # Fill the missing observations
            X.loc[missing_age, 'Age'] = self.model.predict(X.loc[missing_age, self.all_features])
            
            if self.fix_preds:
                
                # Fix for Mr: Socially transitioned to adulthood/Mr around 14
                mask_mr = missing_age & (X['Title'] == 'Mr') & (X['Age'] < 14)
                
                X.loc[mask_mr, 'Age'] = 14
    
                # Fix for Mrs: Almost certainly at least 15 given 1912 marriage norms
                mask_mrs = missing_age & (X['Title'] == 'Mrs') & (X['Age'] < 15)
                
                X.loc[mask_mrs, 'Age'] = 15
    
                # Fix for Master: Only used for boys, typically under 14
                mask_master = missing_age & (X['Title'] == 'Master') & (X['Age'] > 14)
                
                X.loc[mask_master,  'Age'] = 14
                
                print(f"Adjusted {mask_mr.sum()} Mr, {mask_mrs.sum()} Mrs, and {mask_master.sum()} Master records.")
            
        return X
  
    
def apply_group_mask(train_df, test_df, pred_probas, lower = 0.2, upper = 0.8, 
                     threshold = 0.5):
    """
    Adjusts predictions based on the survival of other group members.
    """
    
    # Ensure we start with a clean Survived column for the merge
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Drop Survived from test if it already exists (prevents the rerun bug)
    if 'Survived' in test_df:
        
        test_df = test_df.drop(columns = ['Survived'])
        
    # Create a combined dataset to identify groups across both sets
    full = pd.concat([train_df, test_df], axis = 0).copy()
    
    # Extract Surname
    full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Surname + Ticket as group ID
    full['GroupID'] = full['Surname'] + '_' + full['Ticket'].astype(str).str[:-1] 

    # Calculate survival stats for each group (using only training data)
    group_stats = full.loc[train_df.index].groupby('GroupID')['Survived'].agg(['count', 'mean'])
    
    # Create the Mask
    test_df['GroupID'] = full.loc[full['Survived'].isna(), 'GroupID']
    test_df['Survived'] = pred_probas
    
    # Join the stats back to the test set
    test_df = test_df.merge(group_stats, on = 'GroupID', how = 'left')
    
    # The "Safe" Survivor Titles
    survivor_titles = ['Mrs', 'Miss', 'Master', 'Ms']
    
    # Create common mask
    mask_common = (test_df['count'] > 0) & (test_df['Title'].isin(survivor_titles))
    
    #  Apply the Post-Processing Heuristic
    # If the group had survivors and everyone in it lived (mean == 1)
    # AND you are a woman or child (Master/Miss/Mrs) -> Force Survive
    mask_survive = (test_df['mean'] == 1) & mask_common & (test_df['Survived'] > lower)
    test_df.loc[mask_survive, 'Survived'] = 1
    
    # If the group had members and everyone in it died (mean == 0)
    # AND you are a woman or child -> Force Perish
    mask_perish = (test_df['mean'] == 0) & mask_common & (test_df['Survived'] < upper)
    test_df.loc[mask_perish, 'Survived'] = 0
    
    # Convert to predictions
    test_df['Survived'] = (test_df['Survived'] > threshold).astype(int)
    
    # Convert probabilites to predictions
    preds = (pred_probas > threshold).astype(int)
    
    # Create override masks
    override_survive = mask_survive & (preds != test_df['Survived'].to_numpy())
    override_perish = mask_perish & (preds != test_df['Survived'].to_numpy()) 
    
    print(f"Group Mask: Overrode {override_survive.sum()} to Survivors and {override_perish.sum()} to Perished.")
    
    return test_df['Survived'].values
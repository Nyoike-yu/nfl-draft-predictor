import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('C:\\Personal projects\\GCI_UoJ\\train.csv')
test = pd.read_csv('C:\\Personal projects\\GCI_UoJ\\test.csv')
target = train['Drafted']
groups = train['Year']

def engineer_base_features(df):
    df = df.copy()
    
    # Basic physical features
    df['BMI'] = df['Weight'] / (df['Height'] ** 2) * 703
    df['Jump_Weight_Ratio'] = df['Vertical_Jump'] / (df['Weight'] + 1e-6)
    df['Broad_Height_Ratio'] = df['Broad_Jump'] / (df['Height'] + 1e-6)
    df['Sprint_per_Weight'] = df['Sprint_40yd'] / (df['Weight'] + 1e-6)
    df['Agility_Score'] = df['Agility_3cone'] + df['Shuttle']
    df['Speed_Composite'] = (df['Sprint_40yd'] * 0.5 + df['Agility_3cone'] * 0.3 + df['Shuttle'] * 0.2)
    df['Power_Composite'] = (df['Vertical_Jump'] * 0.5 + df['Broad_Jump'] * 0.3 + df['Bench_Press_Reps'] * 0.2)
    df['Sprint_Agility_Diff'] = df['Sprint_40yd'] - df['Agility_3cone']
    df['Vert_Broad_Ratio'] = df['Vertical_Jump'] / (df['Broad_Jump'] + 1)
    df['Agility_Shuttle_Ratio'] = df['Agility_3cone'] / (df['Shuttle'] + 0.001)
    df['Bench_Weight_Ratio'] = df['Bench_Press_Reps'] / (df['Weight'] + 1e-6)
    df['Jump_Sprint_Ratio'] = df['Broad_Jump'] / (df['Sprint_40yd'] + 0.001)
    df['Height_Weight_Ratio'] = df['Height'] / (df['Weight'] + 1e-6)
    df['Sprint_sq'] = df['Sprint_40yd'] ** 2
    df['Vert_sq'] = df['Vertical_Jump'] ** 2
    df['Broad_sq'] = df['Broad_Jump'] ** 2
    
    drill_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in ['Age'] + drill_cols:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
    df['missing_drills'] = df[drill_cols].isnull().sum(axis=1)
    df['drill_participation_rate'] = 1 - (df['missing_drills'] / len(drill_cols))
    
    pos_group = {
        'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE',
        'FB': 'RB', 'OG': 'OL', 'OT': 'OL', 'C': 'OL',
        'DE': 'DL', 'DT': 'DL', 'OLB': 'LB', 'ILB': 'LB',
        'CB': 'DB', 'SS': 'DB', 'FS': 'DB', 'S': 'DB', 'DB': 'DB',
        'K': 'ST', 'P': 'ST', 'LS': 'ST'
    }
    df['Position_Group'] = df['Position'].map(pos_group)
    return df

# 1. Base Features
train_feat = engineer_base_features(train)
test_feat = engineer_base_features(test)

# School Frequency Encoding 
school_counts = pd.concat([train_feat['School'], test_feat['School']]).value_counts()
train_feat['School_Freq'] = train_feat['School'].map(school_counts)
test_feat['School_Freq'] = test_feat['School'].map(school_counts)

# Positional Z-Scores 
drill_cols = ['Sprint_40yd', 'Vertical_Jump', 'Broad_Jump', 'Bench_Press_Reps', 'Agility_3cone', 'Shuttle']
for col in drill_cols:
    pos_stats = train_feat.groupby('Position')[col].agg(['mean', 'std'])
    train_feat[f'{col}_pos_mean'] = train_feat['Position'].map(pos_stats['mean'])
    test_feat[f'{col}_pos_mean'] = test_feat['Position'].map(pos_stats['mean'])
    global_std = train_feat[col].std()
    train_feat[f'{col}_pos_std'] = train_feat['Position'].map(pos_stats['std']).fillna(global_std)
    test_feat[f'{col}_pos_std'] = test_feat['Position'].map(pos_stats['std']).fillna(global_std)
    train_feat[f'{col}_pos_zscore'] = (train_feat[col] - train_feat[f'{col}_pos_mean']) / (train_feat[f'{col}_pos_std'] + 1e-5)
    test_feat[f'{col}_pos_zscore'] = (test_feat[col] - test_feat[f'{col}_pos_mean']) / (test_feat[f'{col}_pos_std'] + 1e-5)

cat_cols = ['Position', 'Position_Group', 'School', 'Player_Type', 'Position_Type']  
for col in cat_cols:
    train_feat[col] = train_feat[col].fillna('Unknown').astype(str)
    test_feat[col] = test_feat[col].fillna('Unknown').astype(str)
    le = LabelEncoder()
    all_vals = pd.concat([train_feat[col], test_feat[col]], axis=0)
    le.fit(all_vals)
    train_feat[col] = le.transform(train_feat[col])
    test_feat[col] = le.transform(test_feat[col])

# Drop structural columns
drop_cols = ['Id', 'Drafted', 'Year']  
feature_cols = [c for c in train_feat.columns if c not in drop_cols]
X = train_feat[feature_cols]
X_test = test_feat[feature_cols]

# Impute
imputer = KNNImputer(n_neighbors=5)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)


# OPTIMIZED MODEL PARAMETERS 
lgb_params = {
    'n_estimators': 2031, 
    'learning_rate': 0.02064351485101507, 
    'num_leaves': 45, 
    'max_depth': 6, 
    'subsample': 0.8780612290836175, 
    'colsample_bytree': 0.3234563572662656, 
    'reg_alpha': 0.0264011451179221, 
    'reg_lambda': 0.4575265285655216, 
    'min_child_samples': 90,
    'verbose': -1, 
    'random_state': 42, 
    'boosting_type': 'gbdt'
}

cat_params = {
    'iterations': 2500, 'learning_rate': 0.03, 'depth': 6,
    'l2_leaf_reg': 7, 'bagging_temperature': 0.8, 'random_strength': 1.2,
    'early_stopping_rounds': 150, 'verbose': False, 'random_seed': 42
}

et_params = {
    'n_estimators': 1200, 'max_depth': 12, 'min_samples_split': 10,
    'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
    'n_jobs': -1
}

gkf = GroupKFold(n_splits=5) 
oof_preds = {model: np.zeros(len(target)) for model in ['lgb', 'cat', 'et']}
test_preds = {model: np.zeros(len(X_test_imp)) for model in ['lgb', 'cat', 'et']}

print("Training 3-model ensemble with leak-proof GroupKFold CV...")

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_imp, target, groups=groups)):
    X_tr, X_val = X_imp.iloc[tr_idx].copy(), X_imp.iloc[val_idx].copy()
    y_tr, y_val = target.iloc[tr_idx], target.iloc[val_idx]
    

    # NO-LEAK TARGET ENCODING
    global_mean = y_tr.mean()
    k = 20
    test_fold = X_test_imp.copy() 
    
    for grp_col in ['Position', 'Position_Group', 'School']:
        tr_temp = X_tr.copy()
        tr_temp['target'] = y_tr
        
        stats = tr_temp.groupby(grp_col)['target'].agg(['mean', 'count'])
        stats['smooth'] = (stats['mean'] * stats['count'] + global_mean * k) / (stats['count'] + k)
        
        X_tr[grp_col + '_target_enc'] = X_tr[grp_col].map(stats['smooth']).fillna(global_mean)
        X_val[grp_col + '_target_enc'] = X_val[grp_col].map(stats['smooth']).fillna(global_mean)
        test_fold[grp_col + '_target_enc'] = test_fold[grp_col].map(stats['smooth']).fillna(global_mean)
    
    # Train Models
    lgb_m = lgb.LGBMClassifier(**lgb_params)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(False)])
    oof_preds['lgb'][val_idx] = lgb_m.predict_proba(X_val)[:, 1]
    test_preds['lgb'] += lgb_m.predict_proba(test_fold)[:, 1] / gkf.n_splits
    
    cat_m = CatBoostClassifier(**cat_params)
    cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_preds['cat'][val_idx] = cat_m.predict_proba(X_val)[:, 1]
    test_preds['cat'] += cat_m.predict_proba(test_fold)[:, 1] / gkf.n_splits
    
    et_m = ExtraTreesClassifier(**et_params)
    et_m.fit(X_tr, y_tr)
    oof_preds['et'][val_idx] = et_m.predict_proba(X_val)[:, 1]
    test_preds['et'] += et_m.predict_proba(test_fold)[:, 1] / gkf.n_splits

    print(f"  Fold {fold+1} completed.")

# Rank Blending
print("\nApplying Rank Transformation...")
oof_lgb_rank = rankdata(oof_preds['lgb']) / len(target)
oof_cat_rank = rankdata(oof_preds['cat']) / len(target)
oof_et_rank = rankdata(oof_preds['et']) / len(target)

test_lgb_rank = rankdata(test_preds['lgb']) / len(test_preds['lgb'])
test_cat_rank = rankdata(test_preds['cat']) / len(test_preds['cat'])
test_et_rank = rankdata(test_preds['et']) / len(test_preds['et'])

blend_X_rank = np.column_stack([oof_lgb_rank, oof_cat_rank, oof_et_rank])
blend_lr_rank = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
blend_lr_rank.fit(blend_X_rank, target)

weights_rank = blend_lr_rank.coef_[0]
weights_rank = np.maximum(weights_rank, 0)  
if weights_rank.sum() > 0:
    weights_rank /= weights_rank.sum()
else:
    weights_rank = np.array([0.33, 0.33, 0.34]) 

print(f"Rank Blending weights: lgb={weights_rank[0]:.3f}, cat={weights_rank[1]:.3f}, et={weights_rank[2]:.3f}")

final_oof_preds_rank = (weights_rank[0] * oof_lgb_rank + weights_rank[1] * oof_cat_rank + weights_rank[2] * oof_et_rank)
final_auc_rank = roc_auc_score(target, final_oof_preds_rank)
print(f"\n---> HONEST RANK-BLENDED AUC: {final_auc_rank:.5f} <---")

# Final test prediction using ranks
test_pred_rank = (weights_rank[0] * test_lgb_rank + weights_rank[1] * test_cat_rank + weights_rank[2] * test_et_rank)
test_pred_rank = np.clip(test_pred_rank, 0, 1)

try:
    sub = pd.read_csv('C:\\Personal projects\\GCI_UoJ\\sample_submission.csv')
    sub['Drafted'] = test_pred_rank
    sub.to_csv('C:\\Personal projects\\GCI_UoJ\\submission.csv', index=False)
    print(f"\nDone. Leak-proof predictions saved to submission.csv")
except Exception as e:
    print("\nError saving file:", e)

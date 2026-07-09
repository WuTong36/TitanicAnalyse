# -*- coding: utf-8 -*-
"""
Titanic 可调参版本 - 支持命令行参数
用法示例:
  python main.py --weight 3.0 --oversample copy --copies 3 --calibrate True --cv 5
参数	        类型	      默认值	说明
--weight	    float	     2.5       一等舱男性样本权重倍数
--oversample	str	         copy	   过采样方式:smote / copy / none
--copies	    int	         2	       复制过采样时的复制倍数
--calibrate	    bool	     True	   是否进行概率校准
--cv	        int	         5	       交叉验证折数
--test_size	    float	     0.2	   测试集比例
--seed	        int	         42	       随机种子
--output_dir	str	        自定义路径	图片保存根目录
使用默认参数:python main.py
"""

import os
import warnings
import argparse
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

# 尝试导入SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ==============================
# 解析命令行参数
# ==============================
parser = argparse.ArgumentParser(description='Titanic 模型训练与评估')
parser.add_argument('--weight', type=float, default=2.5,
                    help='一等舱男性样本权重倍数 (默认: 2.5)')
parser.add_argument('--oversample', type=str, default='copy', choices=['smote', 'copy', 'none'],
                    help='过采样方法: smote, copy, none (默认: copy)')
parser.add_argument('--copies', type=int, default=2,
                    help='当使用copy过采样时，复制的倍数 (默认: 2)')
parser.add_argument('--calibrate', type=lambda x: x.lower() == 'true', default=True,
                    help='是否进行概率校准 (True/False, 默认: True)')
parser.add_argument('--cv', type=int, default=5,
                    help='交叉验证折数 (默认: 5)')
parser.add_argument('--test_size', type=float, default=0.2,
                    help='测试集比例 (默认: 0.2)')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子 (默认: 42)')
parser.add_argument('--output_dir', type=str, default=r'D:/My-openCV-Project/TitanicAnalyse/Analysis_Output',
                    help='图片输出根目录 (默认: D:/My-openCV-Project/TitanicAnalyse/Analysis_Output)')
args = parser.parse_args()

# 打印当前参数配置
print("="*50)
print("当前参数配置:")
for key, value in vars(args).items():
    print(f"  {key}: {value}")
print("="*50)

# ==============================
# 设置输出目录和日期
# ==============================
today = datetime.now().strftime('%Y%m%d')
base_dir = args.output_dir
fig_dir = os.path.join(base_dir, today)
os.makedirs(fig_dir, exist_ok=True)
print(f"图片保存至: {fig_dir}")

def save_and_show(fig, filename, title_suffix=''):
    full_path = os.path.join(fig_dir, f"{filename}_{today}.png")
    if title_suffix:
        fig.suptitle(f"{fig._suptitle.get_text()} ({today})", fontsize=16)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {full_path}")

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)

# ==============================
# 1. 加载数据
# ==============================
df = pd.read_csv(r'D:/My-openCV-Project/TitanicAnalyse/Titanic-Dataset.csv')
print("数据集大小:", df.shape)
if 'Survived' not in df.columns:
    raise ValueError("缺少 'Survived' 列。")

# ==============================
# 2. EDA（两张图）
# ==============================
print("\n=== 生成EDA图 ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Titanic EDA Combined', fontsize=16)
sns.countplot(x='Survived', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Survival Count')
sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Survival Rate by Sex')
sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Survival Rate by Pclass')
sns.histplot(df['Age'].dropna(), bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Age Distribution')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_and_show(fig, 'EDA_Combined', title_suffix=True)

plt.figure(figsize=(10, 8))
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title(f'Correlation Heatmap ({today})')
plt.savefig(os.path.join(fig_dir, f'Correlation_Heatmap_{today}.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: {os.path.join(fig_dir, f'Correlation_Heatmap_{today}.png')}")

# ==============================
# 3. 特征工程（同之前）
# ==============================
df_clean = df.copy()
df_clean.drop('Cabin', axis=1, inplace=True)
df_clean.loc[:, 'Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean.loc[:, 'Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
df_clean.loc[:, 'Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())

df_clean['Surname'] = df_clean['Name'].apply(lambda x: x.split(',')[0].strip())
df_clean['Title'] = df_clean['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
rare_list = ['Lady', 'Countess', 'Dona', 'Mlle', 'Mme', 'Ms', 'Capt', 'Don', 'Jonkheer']
df_clean['Title'] = df_clean['Title'].apply(lambda x: x if x not in rare_list else 'Rare')
df_clean['Ticket_Prefix'] = df_clean['Ticket'].apply(lambda x: x.split()[0] if len(x.split()) > 1 else 'ONLYNUM')
df_clean['Ticket_Prefix'] = df_clean['Ticket_Prefix'].apply(lambda x: 'ONLYNUM' if x.isdigit() else x)

df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
df_clean['Fare_Pclass_Pct'] = df_clean.groupby('Pclass')['Fare'].rank(pct=True)
df_clean['Fare_Pclass_Bin'] = pd.cut(df_clean['Fare_Pclass_Pct'], bins=5, labels=['P1','P2','P3','P4','P5'])

df_clean['Fare_Per_Person'] = df_clean['Fare'] / df_clean['FamilySize']
df_clean['Is_Head_of_Family'] = ((df_clean['Age'] > 40) & (df_clean['Sex'] == 'male') & (df_clean['FamilySize'] > 1)).astype(int)
df_clean['HighFare_Mr'] = ((df_clean['Fare_Pclass_Bin'].isin(['P4','P5'])) & (df_clean['Title'] == 'Mr')).astype(int)
df_clean['FamilySize_Pclass'] = df_clean['FamilySize'] * df_clean['Pclass']
df_clean['Age_FamilySize'] = df_clean['Age'] * df_clean['FamilySize']
df_clean['Fare_Per_Person_Pclass'] = df_clean['Fare_Per_Person'] * df_clean['Pclass']
df_clean['Head_Fare'] = df_clean['Is_Head_of_Family'] * df_clean['Fare_Per_Person']
df_clean['Pclass_Sex_Alone'] = df_clean['Pclass'].astype(str) + '_' + df_clean['Sex'] + '_' + df_clean['IsAlone'].astype(str)

df_clean.drop(['Name', 'PassengerId'], axis=1, inplace=True)

X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

# ==============================
# 4. 安全构建家族幸存者特征（仅用训练集）
# ==============================
train_temp = X_train.copy()
train_temp['Survived'] = y_train
train_temp['Family_Group'] = train_temp['Surname'] + '_' + train_temp['Ticket_Prefix']
family_size = train_temp.groupby('Family_Group')['Survived'].count().to_dict()
family_survived_sum = train_temp.groupby('Family_Group')['Survived'].sum().to_dict()

def add_family_features(df, family_size, family_survived_sum):
    df = df.copy()
    df['Family_Group'] = df['Surname'] + '_' + df['Ticket_Prefix']
    df['FamilyGroupSize'] = df['Family_Group'].map(family_size).fillna(1).astype(int)
    df['FamilySurvivor_Others'] = df['Family_Group'].map(family_survived_sum).fillna(0).astype(int)
    return df

X_train = add_family_features(X_train, family_size, family_survived_sum)
X_test = add_family_features(X_test, family_size, family_survived_sum)

X_train['FamilySurvivor_Others'] = X_train['FamilySurvivor_Others'] - y_train.values
X_train['HasFamilySurvivor'] = (X_train['FamilySurvivor_Others'] > 0).astype(int)
X_test['HasFamilySurvivor'] = (X_test['FamilySurvivor_Others'] > 0).astype(int)

drop_cols = ['Surname', 'Ticket_Prefix', 'Family_Group', 'FamilySurvivor_Others']
for col in drop_cols:
    if col in X_train.columns: X_train.drop(col, axis=1, inplace=True)
    if col in X_test.columns: X_test.drop(col, axis=1, inplace=True)
X_train.drop('Ticket', axis=1, inplace=True)
X_test.drop('Ticket', axis=1, inplace=True)

def add_agebin(df):
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=['Child','Teenager','Young Adult','Adult','Senior'])
    return df
X_train = add_agebin(X_train)
X_test = add_agebin(X_test)

categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeBin', 'Fare_Pclass_Bin', 'Pclass_Sex_Alone']
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print(f"最终训练特征数: {X_train.shape[1]}")

# ==============================
# 5. 标准化 + 过采样
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 处理NaN
if np.isnan(X_train_scaled).any():
    imputer = SimpleImputer(strategy='median')
    X_train_scaled = imputer.fit_transform(X_train_scaled)
if np.isnan(y_train).any():
    clean_idx = ~np.isnan(y_train)
    X_train_scaled = X_train_scaled[clean_idx]
    y_train = y_train[clean_idx]

male_1st_train = (X_train['Sex_male'] == 1) & (X_train['Pclass'] == 1)
print(f"训练集中一等舱男性样本数: {male_1st_train.sum()}")

# 根据参数选择过采样方式
use_sample_weight = False
if args.oversample == 'smote':
    if HAS_SMOTE:
        print("使用SMOTE过采样...")
        sm = SMOTE(random_state=args.seed, sampling_strategy='auto')
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    else:
        print("SMOTE未安装，回退到copy过采样...")
        args.oversample = 'copy'
if args.oversample == 'copy':
    print(f"使用简单复制过采样（复制{args.copies}倍）...")
    idx = np.where(male_1st_train)[0]
    if len(idx) == 0:
        print("训练集中无一等舱男性，跳过复制。")
        X_train_res = X_train_scaled
        y_train_res = y_train
    else:
        extra_indices = np.tile(idx, args.copies)
        X_train_res = np.vstack([X_train_scaled, X_train_scaled[extra_indices]])
        y_train_res = np.hstack([y_train.values, y_train.iloc[extra_indices].values])
        base_weights = np.ones(len(y_train))
        base_weights[male_1st_train] = args.weight
        extra_weights = np.tile(base_weights[male_1st_train], args.copies)
        sample_weights = np.hstack([base_weights, extra_weights])
        use_sample_weight = True
elif args.oversample == 'none':
    print("不使用过采样。")
    X_train_res = X_train_scaled
    y_train_res = y_train
else:
    raise ValueError("不支持的过采样方法")

print(f"训练集大小: {X_train_res.shape[0]}")

# ==============================
# 6. 交叉验证（使用原始训练集，不经过采样以保持客观）
# ==============================
print(f"\n=== {args.cv}折交叉验证评估 ===")
skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
cv_acc_scores = []
cv_male_errors = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_tr_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    scaler_fold = StandardScaler()
    X_tr_scaled = scaler_fold.fit_transform(X_tr_fold)
    X_val_scaled_fold = scaler_fold.transform(X_val_fold)
    
    model_fold = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=args.seed)
    model_fold.fit(X_tr_scaled, y_tr_fold)
    y_pred_fold = model_fold.predict(X_val_scaled_fold)
    acc_fold = accuracy_score(y_val_fold, y_pred_fold)
    cv_acc_scores.append(acc_fold)
    
    male_mask = (X_val_fold['Sex_male'] == 1) & (X_val_fold['Pclass'] == 1)
    if male_mask.sum() > 0:
        male_y_true = y_val_fold[male_mask]
        male_y_pred = y_pred_fold[male_mask]
        male_err = (male_y_true != male_y_pred).sum() / len(male_y_true) * 100
    else:
        male_err = np.nan
    cv_male_errors.append(male_err)

print(f"各折整体准确率: {cv_acc_scores}")
print(f"平均整体准确率: {np.mean(cv_acc_scores):.4f}")
print(f"各折一等舱男性错误率: {cv_male_errors}")
print(f"平均一等舱男性错误率: {np.nanmean(cv_male_errors):.2f}%")

# ==============================
# 7. 随机森林最终训练（使用全部训练集 + 过采样）
# ==============================
print("\n=== 随机森林超参数调优 ===")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10, 12],
    'min_samples_split': [5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf = RandomForestClassifier(random_state=args.seed)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
if use_sample_weight:
    grid_search.fit(X_train_res, y_train_res, sample_weight=sample_weights)
else:
    grid_search.fit(X_train_res, y_train_res)
best_rf = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}")

# ==============================
# 8. 概率校准（可选）
# ==============================
if args.calibrate:
    print("\n进行概率校准...")
    calibrated_rf = CalibratedClassifierCV(best_rf, method='sigmoid', cv=3)
    calibrated_rf.fit(X_train_res, y_train_res)
    final_model = calibrated_rf
else:
    final_model = best_rf

y_pred_test = final_model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_pred_test)
print(f"测试集准确率 (校准后): {acc_test:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_test))

# 图3: 混淆矩阵
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix ({today})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(fig_dir, f'Confusion_Matrix_{today}.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: {os.path.join(fig_dir, f'Confusion_Matrix_{today}.png')}")

# ==============================
# 9. 一等舱男性专项分析 + 错误率分组图（图4）
# ==============================
print("\n=== 一等舱男性专项分析 (测试集) ===")
if 'Sex_male' in X_test.columns and 'Pclass' in X_test.columns:
    male_1st_mask = (X_test['Sex_male'] == 1) & (X_test['Pclass'] == 1)
    male_1st_X = X_test[male_1st_mask]
    male_1st_y_true = y_test[male_1st_mask]
    male_1st_y_pred = final_model.predict(X_test_scaled[male_1st_mask])
    
    if len(male_1st_X) > 0:
        correct = (male_1st_y_true == male_1st_y_pred).sum()
        errors = len(male_1st_X) - correct
        error_rate_test = errors / len(male_1st_X) * 100
        print(f"测试集一等舱男性错误率: {error_rate_test:.2f}%")
        
        error_df_test = X_test.copy()
        error_df_test['True'] = y_test.values
        error_df_test['Pred'] = y_pred_test
        error_df_test['Wrong'] = (error_df_test['True'] != error_df_test['Pred'])
        error_by_group = error_df_test.groupby(['Sex_male', 'Pclass']).agg(
            total=('Wrong', 'count'),
            errors=('Wrong', 'sum'),
            error_rate=('Wrong', lambda x: x.sum() / x.count() * 100)
        ).reset_index()
        
        plt.figure(figsize=(10, 6))
        error_by_group['Sex_label'] = error_by_group['Sex_male'].map({True: 'Male', False: 'Female'})
        sns.barplot(x='Pclass', y='error_rate', hue='Sex_label', data=error_by_group)
        plt.title(f'Error Rate by Sex & Pclass ({today})')
        plt.xlabel('Pclass')
        plt.ylabel('Error Rate (%)')
        plt.ylim(0, 100)
        plt.legend(title='Sex')
        plt.savefig(os.path.join(fig_dir, f'Error_Rate_By_Group_{today}.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"已保存: {os.path.join(fig_dir, f'Error_Rate_By_Group_{today}.png')}")

# ==============================
# 10. 特征重要性图（图5）
# ==============================
feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_rf.feature_importances_}).sort_values('Importance', ascending=False).head(15)
print("\nTop 15 重要特征:")
print(feat_imp)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title(f'Feature Importance ({today})')
plt.savefig(os.path.join(fig_dir, f'Feature_Importance_{today}.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: {os.path.join(fig_dir, f'Feature_Importance_{today}.png')}")

# ==============================
# 11. 最终结果汇总
# ==============================
print("\n" + "="*50)
print("最终结果汇总:")
print(f"- 参数配置: {vars(args)}")
print(f"- 测试集准确率: {acc_test:.4f}")
print(f"- 测试集一等舱男性错误率: {error_rate_test:.2f}%")
print(f"- 交叉验证平均整体准确率: {np.mean(cv_acc_scores):.4f}")
print(f"- 交叉验证平均一等舱男性错误率: {np.nanmean(cv_male_errors):.2f}%")
print(f"所有五张图片已保存至: {fig_dir}")
print("="*50)
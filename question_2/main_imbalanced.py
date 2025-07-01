import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# 读取数据
df_train = pd.read_csv('question_2\data\cumsum_history.csv')
X = df_train[['user_id_enc', 'blogger_id_enc', '浏览', '点赞', '评论','cumsum_view', 'cumsum_like', 'cumsum_comment']]
y = df_train['label']

# 交叉验证设置
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 记录各折指标
f1_list, auc_list, auprc_list = [], [], []

print("=== 开始 5 折交叉验证（XGBoost + SMOTE）===\n")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"[Fold {fold}]")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # === SMOTE 处理，仅在训练集上做
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # === 模型定义（不用scale_pos_weight了，因为我们已经平衡了）
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=900,
        max_depth=8,
        learning_rate=0.02,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # 训练
    model.fit(X_res, y_res)
    
    # 验证
    y_valid_prob = model.predict_proba(X_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)
    
    # 评估
    f1 = f1_score(y_valid, y_valid_pred)
    auc = roc_auc_score(y_valid, y_valid_prob)
    auprc = average_precision_score(y_valid, y_valid_prob)

    f1_list.append(f1)
    auc_list.append(auc)
    auprc_list.append(auprc)
    
    print(classification_report(y_valid, y_valid_pred, digits=4))
    print(f"Fold {fold} - F1: {f1:.4f} | AUC: {auc:.4f} | AUPRC: {auprc:.4f}\n")

# 汇总
print("=== 5 折交叉验证平均结果 ===")
print(f"Avg F1-score : {np.mean(f1_list):.4f}")
print(f"Avg ROC AUC  : {np.mean(auc_list):.4f}")
print(f"Avg PR AUC   : {np.mean(auprc_list):.4f}")

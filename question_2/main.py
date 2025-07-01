import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve
)



df_train = pd.read_csv('data/train_encoded.csv')
df_test = pd.read_csv('data/test_encoded.csv')

# 1.data properation
X_train = df_train[['user_id_enc', 'blogger_id_enc', '浏览', '点赞', '评论']]
y_train = df_train['label']

X_test = df_test[['user_id_enc', 'blogger_id_enc', '浏览', '点赞', '评论']]

# step 2 build up lightGBM model
# model = lgb.LGBMClassifier(
#     objective='binary',
#     class_weight='balanced', 
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.01,
#     random_state=42
# )

# model.fit(X_train, y_train)

# try xgboost

# 建模：使用类别不平衡时推荐设置 scale_pos_weight
# 计算类别权重比例：负样本数 / 正样本数
# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_estimators=500,
#     max_depth=8,
#     learning_rate=0.01,
#     scale_pos_weight=scale_pos_weight,  # 处理类别不平衡
#     eval_metric='logloss',
#     random_state=42
# )

# # 训练模型
# model.fit(X_train, y_train)


# # step 3 training
# y_train_pred = model.predict(X_train)
# y_train_prob = model.predict_proba(X_train)[:, 1]  # 概率，用于 ROC

# print("训练集评估报告：")
# print(classification_report(y_train, y_train_pred, digits=4))
# print(f"AUC: {roc_auc_score(y_train, y_train_prob):.4f}")


# 2. 类别不平衡权重
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"[INFO] scale_pos_weight = {scale_pos_weight:.2f}")

# 3. 定义模型
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,
    max_depth=8,
    learning_rate=0.01,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 获取训练集预测概率
y_train_prob = model.predict_proba(X_train)[:, 1]

# 6. 使用默认阈值 0.5 预测
y_train_pred_default = (y_train_prob >= 0.5).astype(int)

print("\n=== 训练集评估（默认阈值 0.5）===")
print(classification_report(y_train, y_train_pred_default, digits=4))
print(f"AUC: {roc_auc_score(y_train, y_train_prob):.4f}")
print(f"AUPRC: {average_precision_score(y_train, y_train_prob):.4f}")

# 7. 搜索最优 F1-score 阈值
precision, recall, thresholds = precision_recall_curve(y_train, y_train_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"\n[INFO] 最佳 F1-score 阈值：{best_thresh:.4f}")

# 8. 重新预测
y_train_pred_best = (y_train_prob >= best_thresh).astype(int)

print("\n=== 训练集评估（最优 F1-score 阈值）===")
print(classification_report(y_train, y_train_pred_best, digits=4))
print(f"AUC: {roc_auc_score(y_train, y_train_prob):.4f}")
print(f"AUPRC: {average_precision_score(y_train, y_train_prob):.4f}")

# 9. 测试集预测（概率+标签）
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_thresh).astype(int)

# 10. 输出测试集预测结果（可选保存）
df_test_result = df_test.copy()
df_test_result['pred_prob'] = y_test_prob
df_test_result['pred_label'] = y_test_pred
df_test_result.to_csv('data/test_pred_with_prob.csv', index=False)

print("\n[INFO] 测试集预测已完成，结果保存在 test_pred_with_prob.csv")
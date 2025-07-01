import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
import joblib

df_train = pd.read_csv('q_2_method2/train_user_blogger_features_encoded.csv')
df_pred = pd.read_csv('q_2_method2/candidate_with_user_and_blogger_features_encoded.csv')

# 定义用于分析的特征列（不包括 label）
feature_cols = [
    'user_id_enc', '浏览', '点赞', '评论',
    'cumsum_view', 'cumsum_like', 'cumsum_comment',
    'total_view', 'total_like', 'total_comment',
    'fan_count', 'user_count', 'fan_ratio'
]

X = df_train[feature_cols]

# 计算特征之间的相关性矩阵
corr_matrix = X.corr(method='pearson').round(2)
print(corr_matrix)


# 设置中文字体（SimHei黑体）和负号正常显示
plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Matrix")
# plt.tight_layout()
# plt.show()


# 输入与标签
X_train = df_train[feature_cols]
y_train = df_train['label']

# 类别不平衡处理
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 初始化模型
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=400,
    learning_rate=0.01,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 训练模型（用训练集自身作为验证集启用早停）
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    verbose=True
)

# 预测概率与类别
y_pred = model.predict(X_train)
y_prob = model.predict_proba(X_train)[:, 1]

# 评估指标
print("=== 训练集评估报告 ===")
print(classification_report(y_train, y_pred, digits=4))
print(f"AUC       : {roc_auc_score(y_train, y_prob):.4f}")
print(f"PR AUC    : {average_precision_score(y_train, y_prob):.4f}")

joblib.dump(model, 'q_2_method2/xgb_model.pkl')
# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(y_train, y_prob)

# 计算 PR 曲线
precision, recall, _ = precision_recall_curve(y_train, y_prob)

# 绘制 ROC 曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', label=f'ROC 曲线 (AUC = {roc_auc_score(y_train, y_prob):.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('ROC 曲线')
plt.legend()
plt.grid(True)

# 绘制 PR 曲线
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', label=f'PR 曲线 (AUC = {average_precision_score(y_train, y_prob):.4f})')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('Precision-Recall 曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

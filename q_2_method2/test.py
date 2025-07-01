import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# === Step 1: 读取候选数据（含原始 User ID 和 Blogger ID） ===
df_pred = pd.read_csv('q_2_method2/candidate_with_user_and_blogger_features_encoded.csv')

# === Step 2: 定义模型使用的特征列 ===
feature_cols = [
    'user_id_enc', '浏览', '点赞', '评论',
    'cumsum_view', 'cumsum_like', 'cumsum_comment',
    'total_view', 'total_like', 'total_comment',
    'fan_count', 'user_count', 'fan_ratio'
]

X_pred = df_pred[feature_cols]

# === Step 3: 加载训练好的模型 ===
model = joblib.load('q_2_method2/xgb_model.pkl')

# === Step 4: 模型预测 ===
df_pred['prob'] = model.predict_proba(X_pred)[:, 1]
df_pred['pred'] = (df_pred['prob'] >= 0.5).astype(int)

# === Step 5: 筛选预测为“新增关注”的记录 ===
df_result = df_pred[df_pred['pred'] == 1]

# === Step 6: 限定指定目标用户 ===
target_users = ['U7', 'U6749', 'U5769', 'U14990', 'U52010']
df_result = df_result[df_result['User ID'].isin(target_users)]

# === Step 7: 可选：每个用户仅保留 Top-K 高概率记录（如 Top 5） ===
df_result = df_result.sort_values(by=['User ID', 'prob'], ascending=[True, False])
df_result = df_result.groupby('User ID').head(5)

# === Step 8: 整理为 User ID → 新关注博主列表 ===
df_summary = df_result.groupby('User ID')['Blogger ID'].apply(list).reset_index()
df_summary['new_followed_bloggers'] = df_summary['Blogger ID'].apply(lambda x: ','.join(map(str, x)))
df_summary = df_summary[['User ID', 'new_followed_bloggers']]



# === Step 9: 保存结果到 CSV 文件 ===
df_summary.to_csv('q_2_method2/final_user_blogger_map.csv', index=False, encoding='utf-8-sig')
print("结果已保存至：q_2_method2/final_user_blogger_map.csv")



# === Step 10: SHAP 分析（模型特征重要性解释） ===

# 创建 SHAP explainer（适用于 tree-based 模型）
explainer = shap.TreeExplainer(model)

# 计算 SHAP 值（仅对前 1000 个样本，避免太慢）
shap_values = explainer.shap_values(X_pred[:1000])

# 设置中文显示（SimHei）和负号
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# === 绘制 SHAP summary plot（特征影响全局排序图） ===
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_pred[:1000], show=False)
plt.tight_layout()
plt.savefig('q_2_method2/shap_summary_plot.png', dpi=300)
plt.close()
print("SHAP summary plot 已保存至：q_2_method2/shap_summary_plot.png")

# === 可选：绘制 top1 特征的 dependence plot（交互关系） ===
top_feature = X_pred.columns[abs(shap_values).mean(0).argmax()]
shap.dependence_plot(top_feature, shap_values, X_pred[:1000], show=False)
plt.tight_layout()
plt.savefig('q_2_method2/shap_dependence_plot_top1.png', dpi=300)
plt.close()
print(f"SHAP dependence plot（{top_feature}）已保存至：q_2_method2/shap_dependence_plot_top1.png")

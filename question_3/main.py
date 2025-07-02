# main.py
import pandas as pd
from data_process import extract_user_features_combined
from model_online import train_online_model, predict_target_online

# 加载数据
df = pd.read_csv('data/day_activity_used_to_train.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 构造周期 + 趋势特征
feature_df = extract_user_features_combined(df)

# 训练模型（自动过滤目标用户）
model, scaler = train_online_model(df, feature_df)

# 预测目标用户在 2024-07-21 是否在线
result = predict_target_online(feature_df)
print("=== 预测结果 ===")
print(result)

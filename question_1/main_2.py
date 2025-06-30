import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

daily_stats = pd.read_csv('data/daily_stats.csv')
train_df = daily_stats.copy
train_df = daily_stats.dropna(subset=['浏览', '点赞', '评论', '关注'])
# 特征与标签
X_train = train_df[['浏览', '点赞', '评论']]
y_train = train_df['关注']

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 构建模型
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 验证集评估
y_val_pred = model.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
rmse = root_mean_squared_error(y_val, y_val_pred)

print(f"验证集 R²: {r2:.4f}")
print(f"验证集 MAE: {mae:.2f}")
print(f"验证集 RMSE: {rmse:.2f}")

# 加载0721预测行为数据
df_test = pd.read_csv('data/predict_0721_behavior.csv')

# 特征列
X_test = df_test[['浏览', '点赞', '评论']]

#  预测关注数并写回 
y_pred = model.predict(X_test)
df_test['关注'] = y_pred.round().astype(int)

# 保存预测结果
df_test.to_csv('data/predict_0721_behavior_with_follow.csv', index=False)
print("✅ 已完成关注数预测，保存为：predict_0721_behavior_with_follow.csv")

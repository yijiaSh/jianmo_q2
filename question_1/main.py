import pandas as pd 
import matplotlib.pyplot as plt
from data_process import daily_stats
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

# ------------------ 配置参数 ------------------
time_steps = 5
epochs = 500
batch_size = 32
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# ------------------ 模型结构 ------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)

# ------------------ 样本构造函数 ------------------
def build_sequence_samples(df, target_col, time_steps=5):
    X_all = []
    y_all = []

    for blogger_id, group in df.groupby('Blogger ID'):
        group = group.sort_values('Date')
        features = group[['浏览', '点赞', '评论', '关注']].values
        target = group[target_col].values

        if len(group) <= time_steps:
            continue

        for i in range(len(group) - time_steps):
            X_all.append(features[i:i+time_steps])
            y_all.append(target[i + time_steps])

    return np.array(X_all), np.array(y_all)

# ------------------ 预测值反归一化 ------------------
def inverse_single_value(scaler, norm_value, col_index):
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, col_index] = norm_value
    return scaler.inverse_transform(dummy)[0, col_index]

# ------------------ 主循环：浏览 / 点赞 / 评论 ------------------
target_list = [
    ('浏览', 0, 'pred_view.csv'),
    ('点赞', 1, 'pred_like.csv'),
    ('评论', 2, 'pred_comment.csv')
]

# 归一化数据
scaler = MinMaxScaler()
scaled_data = daily_stats.copy()
scaled_data[['浏览', '点赞', '评论', '关注']] = scaler.fit_transform(scaled_data[['浏览', '点赞', '评论', '关注']])

for target_col, col_index, filename in target_list:
    print(f'\n 开始处理目标列：{target_col}')

    # 构造样本
    X, y = build_sequence_samples(scaled_data, target_col=target_col, time_steps=time_steps)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 构造模型
    model = LSTMRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 训练模型
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss = {total_loss:.2f}")

    # 预测0721当天
    model.eval()
    predict_result = []

    for blogger_id, group in scaled_data.groupby('Blogger ID'):
        group = group.sort_values('Date')
        recent_seq = group[['浏览', '点赞', '评论', '关注']].values[-time_steps:]

        if recent_seq.shape[0] < time_steps:
            continue

        x_input = torch.tensor(recent_seq, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
        pred = model(x_input).item()
        predict_result.append((blogger_id, pred))

    df_pred = pd.DataFrame(predict_result, columns=['Blogger ID', f'预测{target_col}_0721'])

    # 反归一化
    df_pred[f'{target_col}_0721_还原'] = df_pred[f'预测{target_col}_0721'].apply(
        lambda x: round(inverse_single_value(scaler, x, col_index))
    ).astype(int)

    # 保存结果
    df_pred.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"已保存 {target_col} 预测结果到 {filename}")

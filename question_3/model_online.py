# model_online.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_online_model(df_all, feature_df, model_path='question_3/online_model.pkl'):
    """
    构建全用户训练集（含标签），用于训练逻辑回归模型
    标签规则：某用户在 2024-07-21 有任意行为 → 在线（1），否则为 0
    """
    # 构建标签（仅从 df_all 中获取）
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    target_date = pd.Timestamp('2024-07-21')
    user_online_set = set(df_all[df_all['Date'] == target_date]['User ID'].unique())

    feature_df['label'] = feature_df['User ID'].apply(lambda uid: 1 if uid in user_online_set else 0)

    # 保存预测目标用户 ID
    target_uids = ['U9', 'U22405', 'U16', 'U48420']
    target_feature_df = feature_df[feature_df['User ID'].isin(target_uids)]

    # 剔除预测目标用户，仅用于测试
    train_df = feature_df[~feature_df['User ID'].isin(target_uids)].copy()

    if train_df['label'].sum() == 0:
        print("⚠️ 训练数据中无正样本（在线用户），请检查样本范围或增加历史数据。")
        return None, target_feature_df

    X_train = train_df.drop(['User ID', 'label'], axis=1)
    y_train = train_df['label']

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # 保存模型 & 标准器
    joblib.dump((model, scaler), model_path)
    print(f"模型已训练并保存至 {model_path}")

    return model, scaler

def predict_target_online(feature_df, model_path='question_3/online_model.pkl'):
    """
    对目标用户预测 07.21 是否在线
    """
    target_uids = ['U9', 'U22405', 'U16', 'U48420']
    df_test = feature_df[feature_df['User ID'].isin(target_uids)].copy()

    if not os.path.exists(model_path):
        print("⚠️ 模型文件不存在，请先训练模型。")
        return df_test

    model, scaler = joblib.load(model_path)

    X_test = df_test.drop(['User ID'], axis=1)
    X_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    df_test['online_0721'] = preds
    df_test['probability'] = probs.round(4)
    return df_test[['User ID', 'online_0721', 'probability']]

import pandas as pd


# 筛选目标用户
## history
df = pd.read_csv('question_2\data\cumsum_history.csv')
target_users = ['U7','U6749','U5769','U14990','U52010']
df_target = df[df['User ID'].isin(target_users)]
df_target = df_target.iloc[:, 2:]
df_target= df_target.drop(columns=['user_id_enc', 'blogger_id_enc'])
df_target.to_csv('data/target_user_history.csv', index=False)
print(df_target)

## 0722
df_0722 = pd.read_csv('data/0722_activity_used_to_train.csv')
print(target_users)
df_target_0722 = df_0722[df_0722['User ID'].isin(target_users)]
df_target_0722 = df_target_0722.iloc[:, 1:]
df_target_0722.to_csv('data/target_user_0722.csv', index=False)
print(df_target_0722)


# 获取用户已关注列表
df_label1 = df_target[df_target['label']==1]
user_blogger_dict = df_label1.groupby('User ID')['Blogger ID'].unique().to_dict()
result_df = pd.DataFrame(list(user_blogger_dict.items()), columns=['User ID', 'Blogger IDs'])

result_df.to_csv('data/user_blogger_mapping.csv', index=False)

# 打印结果
print("User ID与Blogger ID的映射字典：")
print(user_blogger_dict)

df_user_hist = pd.read_csv('data/target_user_history.csv')

df_all = pd.read_csv('question_2/data/cumsum_history.csv')
# --- 确保日期格式正确（仅当需要过滤时用）
df_all['Date'] = pd.to_datetime(df_all['Date'])

# 3. 构建博主画像特征（全体用户维度）

# 3.1 博主被互动总量
df_blog_interact = df_all.groupby('Blogger ID')[['浏览', '点赞', '评论']].sum().rename(
    columns={'浏览': 'total_view', '点赞': 'total_like', '评论': 'total_comment'}
).reset_index()

# 3.2 博主被关注次数（label=1）
df_blog_fan = df_all[df_all['label'] == 1].groupby('Blogger ID').size().reset_index(name='fan_count')

# 3.3 博主被访问用户数（user_count）
df_blog_user = df_all.groupby('Blogger ID')['User ID'].nunique().reset_index(name='user_count')

# 3.4 合并博主侧画像
df_blogger_stats = df_blog_interact.merge(df_blog_fan, on='Blogger ID', how='left')
df_blogger_stats = df_blogger_stats.merge(df_blog_user, on='Blogger ID', how='left')

df_blogger_stats['fan_count'] = df_blogger_stats['fan_count'].fillna(0)
df_blogger_stats['user_count'] = df_blogger_stats['user_count'].fillna(1)
df_blogger_stats['fan_ratio'] = df_blogger_stats['fan_count'] / df_blogger_stats['user_count']

# 4. 合并博主画像进用户历史行为记录
df_user_hist_with_blogger = df_user_hist.merge(
    df_blogger_stats[['Blogger ID', 'total_view', 'total_like', 'total_comment', 'fan_count', 'user_count', 'fan_ratio']],
    on='Blogger ID',
    how='left'
)

# 若有个别博主画像缺失，统一填 0
df_user_hist_with_blogger[['total_view', 'total_like', 'total_comment', 'fan_count', 'user_count', 'fan_ratio']] = \
    df_user_hist_with_blogger[['total_view', 'total_like', 'total_comment', 'fan_count', 'user_count', 'fan_ratio']].fillna(0)

# 5. 保存为新的训练数据
df_user_hist_with_blogger.to_csv('data/train_user_blogger_features.csv', index=False)

print("用户训练集已补齐博主画像并保存为：data/train_user_blogger_features.csv")


# 最后处理两个文件的映射



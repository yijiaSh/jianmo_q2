import pandas as pd

# 重新读取0722行为数据（含点赞/浏览/评论等）
df_target_0722 = pd.read_csv('data/target_user_0722.csv')
# 读取用户已关注博主映射
user_blogger_map = pd.read_csv('data/user_blogger_mapping.csv')
user_blogger_map['Blogger IDs'] = user_blogger_map['Blogger IDs'].apply(eval)
user_blogger_dict = dict(zip(user_blogger_map['User ID'], user_blogger_map['Blogger IDs']))

# 判断是否已关注
def check_followed(row):
    return row['Blogger ID'] in user_blogger_dict.get(row['User ID'], [])

df_target_0722['already_followed'] = df_target_0722.apply(check_followed, axis=1)

# 筛选“未关注但0722有互动”的候选记录
df_candidate = df_target_0722[(df_target_0722['already_followed'] == False) &
                              (df_target_0722[['浏览', '点赞', '评论']].sum(axis=1) > 0)]

df_candidate.to_csv('data/candidate_0722.csv', index=False)

print("目标用户0722未关注但有互动的候选博主：")
print(df_candidate[['User ID', 'Blogger ID', '浏览', '点赞', '评论']])



# 加载完整历史数据（包含累计行为字段）
df_all = pd.read_csv('question_2/data/cumsum_history.csv')
df_all['Date'] = pd.to_datetime(df_all['Date'])

target_users = ['U7','U6749','U5769','U14990','U52010']
df_hist = df_all[df_all['User ID'].isin(target_users)].copy()

# 只保留0722之前的数据（含0720）
df_hist = df_hist[df_hist['Date'] <= '2024-07-20']

# 按用户-博主统计截止0722之前的累计行为
df_cumsum_up_to_0722 = df_hist.groupby(['User ID', 'Blogger ID'])[['浏览', '点赞', '评论']].sum().reset_index()
df_cumsum_up_to_0722 = df_cumsum_up_to_0722.rename(columns={
    '浏览': 'cumsum_view',
    '点赞': 'cumsum_like',
    '评论': 'cumsum_comment'
})


df_candidate = df_candidate.merge(
    df_cumsum_up_to_0722,
    on=['User ID', 'Blogger ID'],
    how='left'
)

# 若某些用户与博主之前没有历史互动，则填0
df_candidate[['cumsum_view', 'cumsum_like', 'cumsum_comment']] = df_candidate[[
    'cumsum_view', 'cumsum_like', 'cumsum_comment'
]].fillna(0)


# 统计博主被互动总量（全体用户）
blogger_interact_stats = df_all.groupby('Blogger ID')[['浏览', '点赞', '评论']].sum().rename(
    columns=lambda x: f'total_{x}'
).reset_index()

# 统计博主被关注次数
blogger_fan_count = df_all[df_all['label'] == 1].groupby('Blogger ID').size().reset_index(name='fan_count')

# 统计博主被多少用户访问过（独立用户数）
blogger_user_count = df_all.groupby('Blogger ID')['User ID'].nunique().reset_index(name='user_count')

# 合并并计算转化率
df_blogger_stats = blogger_interact_stats.merge(blogger_fan_count, on='Blogger ID', how='left')
df_blogger_stats = df_blogger_stats.merge(blogger_user_count, on='Blogger ID', how='left')

df_blogger_stats['fan_count'] = df_blogger_stats['fan_count'].fillna(0)
df_blogger_stats['user_count'] = df_blogger_stats['user_count'].fillna(1)
df_blogger_stats['fan_ratio'] = df_blogger_stats['fan_count'] / df_blogger_stats['user_count']

# 合并博主统计画像
df_candidate_final = df_candidate.merge(df_blogger_stats, on='Blogger ID', how='left')

# 选定模型特征列（可灵活增减）
feature_cols = [
    'User ID', 'Blogger ID',
    '浏览', '点赞', '评论',
    'cumsum_view', 'cumsum_like', 'cumsum_comment',
    'total_浏览', 'total_点赞', 'total_评论',
    'fan_count', 'fan_ratio'
]

# 打印查看
print(df_candidate_final[feature_cols].head())

# 保存为建模输入文件
df_candidate_final.to_csv('data/candidate_with_user_and_blogger_features.csv', index=False)


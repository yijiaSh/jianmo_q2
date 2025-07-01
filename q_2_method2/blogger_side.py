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
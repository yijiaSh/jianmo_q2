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



import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 读取原始行为数据
# df = pd.read_csv('data/Attachment_2.csv')  # 包含 User ID, User behaviour, Blogger ID, Time

# # 转换时间格式并提取日期
# df['Time'] = pd.to_datetime(df['Time'])
# df['Date'] = df['Time'].dt.date  

# # 按 (User ID, Blogger ID, Date, 行为类型) 聚合行为次数
# df_grouped = df.groupby(['User ID', 'Blogger ID', 'Date', 'User behaviour']).size().unstack(fill_value=0).reset_index()

# # 确保所有行为列存在
# for col in [1, 2, 3, 4]:  # 1=浏览, 2=点赞, 3=评论, 4=关注
#     if col not in df_grouped.columns:
#         df_grouped[col] = 0

# # 添加 label 列
# df_grouped['label'] = df_grouped[4].apply(lambda x: 1 if x > 0 else 0)

# # df_grouped.to_csv('data/0722_activity.csv')

# # 删除“关注”行为列，避免泄露
# df_grouped = df_grouped.drop(columns=[4])

# # 重命名列为可读形式
# df_grouped = df_grouped.rename(columns={1: '浏览', 2: '点赞', 3: '评论'})

# # df_grouped.to_csv('data/0722_activity_used_to_train.csv')

# # 查看结果
# print(df_grouped.head)



# 对用户和博主ID进行一致性编码，.fit,LabelEncoder
df_train = pd.read_csv('data/day_activity_used_to_train.csv')
df_test = pd.read_csv('data/0722_activity_used_to_train.csv')

all_user_ids = pd.concat([df_train['User ID'], df_test['User ID']]).unique()
all_blogger_ids = pd.concat([df_train['Blogger ID'], df_test['Blogger ID']]).unique()

# 初始化编码器并 fit 全体 ID
user_encoder = LabelEncoder()
blogger_encoder = LabelEncoder()

user_encoder.fit(all_user_ids)
blogger_encoder.fit(all_blogger_ids)

# 编码训练集
df_train['user_id_enc'] = user_encoder.transform(df_train['User ID'])
df_train['blogger_id_enc'] = blogger_encoder.transform(df_train['Blogger ID'])

# 编码测试集
df_test['user_id_enc'] = user_encoder.transform(df_test['User ID'])
df_test['blogger_id_enc'] = blogger_encoder.transform(df_test['Blogger ID'])

print(user_encoder.classes_[:5])  # ['U1', 'U10', 'U100', ...]
print(blogger_encoder.classes_[:5])


# df_train.to_csv('data/train_encoded.csv', index=False)
# df_test.to_csv('data/test_encoded.csv', index=False)


# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('data/train_encoded.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 按日期和标签统计数量
# count_df = df.groupby(['Date', 'label']).size().reset_index(name='count')

# # 创建透视表
# pivot_df = count_df.pivot(index='Date', columns='label', values='count').fillna(0)
# pivot_df.columns = ['未关注', '关注']

# # 绘图
# ax = pivot_df.plot(kind='bar', figsize=(12, 6), color=['gray', 'orange'])

# # 添加数值标签
# for container in ax.containers:
#     ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9, padding=3)

# # 图形设置
# plt.title('每日关注/未关注样本数量统计')
# plt.xlabel('日期')
# plt.ylabel('样本数量')
# plt.xticks(rotation=45)
# plt.legend(title='标签')
# plt.tight_layout()
# plt.show()

# 按 user + blogger + 日期 升序排列
df = df.sort_values(by=['user_id_enc', 'blogger_id_enc', 'Date'])

# 累计浏览、点赞、评论（包含当日）
df['cumsum_view'] = df.groupby(['user_id_enc', 'blogger_id_enc'])['浏览'].cumsum()
df['cumsum_like'] = df.groupby(['user_id_enc', 'blogger_id_enc'])['点赞'].cumsum()
df['cumsum_comment'] = df.groupby(['user_id_enc', 'blogger_id_enc'])['评论'].cumsum()

# 示例查看
print(df[['user_id_enc', 'blogger_id_enc', 'Date', '浏览', '点赞', '评论',
          'cumsum_view', 'cumsum_like', 'cumsum_comment']].head(10))

df.to_csv('question_2/data/cumsum_history.csv')





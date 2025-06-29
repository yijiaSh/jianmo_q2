import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data/Attachment_1.csv')
# view how many bloggers
# sorted_ids = sorted(df['Blogger ID'].unique())
# print(sorted_ids)


df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = df['Time'].dt.date


behavior_map = {'1': '浏览', '2': '点赞', '3': '评论', '4': '关注'}
df['User behaviour'] = df['User behaviour'].astype(str).map(behavior_map)


daily_stats = (
    df.groupby(['Blogger ID', 'Date', 'User behaviour'])
      .size()
      .unstack(fill_value=0)
      .reset_index()  
)

print(daily_stats)

daily_stats.to_csv('data/daily_stats.csv')

# show ever blogger's number of seeing, like, mark. follow
# set fort
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False



# for blogger_id, group in daily_stats.groupby('Blogger ID'):
#     plt.figure(figsize=(10, 6))
#     group = group.sort_values('Date')

#     plt.plot(group['Date'], group['浏览'], marker='o', label='浏览')
#     plt.plot(group['Date'], group['点赞'], marker='s', label='点赞')
#     plt.plot(group['Date'], group['评论'], marker='^', label='评论')
#     plt.plot(group['Date'], group['关注'], marker='*', label='关注')

#     plt.title(f'博主 {blogger_id} 的用户行为趋势')
#     plt.xlabel('日期')
#     plt.ylabel('行为次数')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.grid(True)
#     plt.savefig(f'question_1/pic/{blogger_id}.png')
#     plt.close
    




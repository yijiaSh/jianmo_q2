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



for blogger_id, group in daily_stats.groupby('Blogger ID'):
    group = group.sort_values('Date')

    plt.figure(figsize=(10, 6))

    # 主 Y 轴（左）—— 浏览、点赞、评论
    ax1 = plt.gca()  # 获取当前坐标轴
    ax1.plot(group['Date'], group['浏览'], marker='o', label='浏览', color='tab:blue')
    ax1.plot(group['Date'], group['点赞'], marker='s', label='点赞', color='tab:orange')
    ax1.plot(group['Date'], group['评论'], marker='^', label='评论', color='tab:green')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('浏览 / 点赞 / 评论 次数')
    ax1.tick_params(axis='y', labelcolor='black')

    # 副 Y 轴（右）—— 关注
    ax2 = ax1.twinx()
    ax2.plot(group['Date'], group['关注'], marker='*', label='关注', color='tab:red', linewidth=2)
    ax2.set_ylabel('关注次数', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 图标题与图例
    plt.title(f'博主 {blogger_id} 的用户行为趋势（双Y轴）')

    # 图例合并（左右轴都参与）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'question_1/pic_2/{blogger_id}.png')
    plt.close()
    




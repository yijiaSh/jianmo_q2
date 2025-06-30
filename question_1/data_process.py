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
    

# 处理预测后的0721 浏览/点赞/评论 数据
# 读取三个预测文件（包含还原后的数据）
df_view = pd.read_csv('data/pred_view.csv')[['Blogger ID', '浏览_0721_还原']]
df_like = pd.read_csv('data/pred_like.csv')[['Blogger ID', '点赞_0721_还原']]
df_comment = pd.read_csv('data/pred_comment.csv')[['Blogger ID', '评论_0721_还原']]

# 合并数据
df_merged = df_view.merge(df_like, on='Blogger ID').merge(df_comment, on='Blogger ID')

# 加上日期列和占位的关注列（等你第二阶段预测后再填）
df_merged['Date'] = '2024-07-21'
df_merged['关注'] = -1  # 暂时占位，后续可替换为预测值

# 调整列顺序为：Blogger ID, Date, 关注, 浏览, 点赞, 评论
df_final = df_merged[['Blogger ID', 'Date', '关注', '浏览量_0721_还原', '点赞量_0721_还原', '评论量_0721_还原']]
df_final.columns = ['Blogger ID', 'Date', '关注', '浏览', '点赞', '评论']  # 重命名列

# 保存最终整合结果
df_final.to_csv('data/predict_0721_behavior.csv', index=False)
print("✅ 已生成 7.21 的行为整合表：predict_0721_behavior.csv")

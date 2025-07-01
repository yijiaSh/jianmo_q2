import pandas as pd


# 处理预测后的0721 浏览/点赞/评论 数据
# 读取三个预测文件（包含还原后的数据）
df_view = pd.read_csv('data/pred_view.csv')[['Blogger ID', '浏览_0721_还原']]
df_like = pd.read_csv('data/pred_like.csv')[['Blogger ID', '点赞_0721_还原']]
df_comment = pd.read_csv('data/pred_comment.csv')[['Blogger ID', '评论_0721_还原']]

# 合并数据
df_merged = df_view.merge(df_like, on='Blogger ID').merge(df_comment, on='Blogger ID')

# 加上日期列和占位的关注列（第二阶段预测后再填）
df_merged['Date'] = '2024-07-21'
df_merged['关注'] = -1  # 暂时占位，后续可替换为预测值

# 调整列顺序为：Blogger ID, Date, 关注, 浏览, 点赞, 评论
df_final = df_merged[['Blogger ID', 'Date', '关注', '浏览_0721_还原', '点赞_0721_还原', '评论_0721_还原']]
df_final.columns = ['Blogger ID', 'Date', '关注', '浏览', '点赞', '评论']  # 重命名列

# 保存最终整合结果
df_final.to_csv('data/predict_0721_behavior.csv', index=False)
print("已生成 7.21 的行为整合表：predict_0721_behavior.csv")



# 对筛选
df = pd.read_csv('data/predict_0721_behavior_with_follow.csv')
top5 = df.sort_values('关注', ascending=False).head(5)
print("0721 预测关注 Top5 博主：")
print(top5[['Blogger ID', '关注']])
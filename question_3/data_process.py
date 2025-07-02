import pandas as pd



def extract_user_features_combined(df_all):
    """
    提取10到20日之间用户互动数据，周行为特征，用于判断用户在0721是否在线

    Args:
        df_all (_type_): _description_
    """
    
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    # 添加互动强度列（点赞 + 评论 + 关注）
    df_all['interaction'] = df_all[['点赞', '评论', '关注']].sum(axis=1)

    # 周期窗口（用于模拟上周行为习惯）
    period_start = pd.Timestamp('2024-07-11')
    period_end = pd.Timestamp('2024-07-17')

    # 趋势窗口（用于捕捉近期活跃趋势）
    trend_start = pd.Timestamp('2024-07-18')
    trend_end = pd.Timestamp('2024-07-20')

    # 所有用户（包括目标用户和其他用户）
    all_users = df_all['User ID'].unique()
    result_rows = []

    for uid in all_users:
        row = {'User ID': uid}

        # ========== 周期性行为特征 ==========
        df_period = df_all[(df_all['Date'] >= period_start) & (df_all['Date'] <= period_end)]
        df_u_period = df_period[df_period['User ID'] == uid].copy()
        df_u_period['is_active'] = df_u_period['interaction'] > 0
        df_u_period_active = df_u_period[df_u_period['is_active']].drop_duplicates(subset='Date')

        row['period_active_days'] = df_u_period_active['Date'].nunique()
        df_u_period_active['weekday'] = df_u_period_active['Date'].dt.weekday
        row['period_weekday_active_days'] = df_u_period_active[df_u_period_active['weekday'] < 5].shape[0]
        row['period_weekend_active_days'] = df_u_period_active[df_u_period_active['weekday'] >= 5].shape[0]
        row['period_was_active_on_sunday'] = int(pd.Timestamp('2024-07-14') in df_u_period_active['Date'].values)

        if not df_u_period_active.empty:
            row['period_recent_gap_to_0717'] = (pd.Timestamp('2024-07-17') - df_u_period_active['Date'].max()).days
        else:
            row['period_recent_gap_to_0717'] = 10  # 最大间隔

        row['period_avg_daily_interaction'] = df_u_period.groupby('Date')['interaction'].sum().mean() or 0
        row['period_total_interaction'] = df_u_period['interaction'].sum()

        # ========== 趋势性行为特征 ==========
        df_trend = df_all[(df_all['Date'] >= trend_start) & (df_all['Date'] <= trend_end)]
        df_u_trend = df_trend[df_trend['User ID'] == uid].copy()
        df_u_trend['is_active'] = df_u_trend['interaction'] > 0
        df_u_trend_active = df_u_trend[df_u_trend['is_active']].drop_duplicates(subset='Date')

        row['trend_active_days'] = df_u_trend_active['Date'].nunique()
        row['trend_avg_daily_interaction'] = df_u_trend.groupby('Date')['interaction'].sum().mean() or 0
        row['trend_total_interaction'] = df_u_trend['interaction'].sum()
        row['trend_was_active_on_0720'] = int(pd.Timestamp('2024-07-20') in df_u_trend_active['Date'].values)

        result_rows.append(row)

    return pd.DataFrame(result_rows)
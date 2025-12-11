def add_time_features(df):
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofweek"] = df.index.dayofweek
    return df

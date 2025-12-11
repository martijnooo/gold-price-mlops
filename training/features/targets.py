def add_target(df, target_col='Gold_Price', shift=-1, new_col='Target_Gold_Price'):
    df[new_col] = df[target_col].shift(shift)
    return df

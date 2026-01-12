import numpy as np

def add_target(df, target_col='Gold_Price', shift=-1, new_col='Target_Gold_Price'):
    df[new_col] = df[target_col].shift(shift)
    # Return Target (Log return to the next period)
    df['Target_Return'] = np.log(df[target_col].shift(shift) / df[target_col])
    return df

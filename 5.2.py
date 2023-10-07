import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('data/titanic_with_labels.csv')
    ind = df[(df['sex'] != 'М') & (df['sex'] != 'Ж')].index
    df.drop(ind, inplace=True)
    df.sex.replace(['М', 'Ж'], ['1', '0'], inplace=True)
    df.row_number.fillna(df.row_number.max(), inplace=True)
    df.loc[df.liters_drunk < 0, 'liters_drunk'] = np.nan
    df.liters_drunk = hampel(df.liters_drunk)
    df.liters_drunk = df.liters_drunk.fillna(df.liters_drunk.mean())
    df.to_csv('data/res.csv')


def hampel(vals_orig):
    vals = vals_orig.copy()
    diff = np.abs(vals.median() - vals)
    mad = diff.median()
    threshold = 3 * mad
    outlier_idx = diff > threshold
    vals[outlier_idx] = np.nan
    return vals


if __name__ == "__main__":
    main()

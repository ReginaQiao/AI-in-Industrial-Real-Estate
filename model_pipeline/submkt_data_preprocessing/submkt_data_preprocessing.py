import os
import warnings
import etl.connect as conn
import pandas as pd
from typing import List, Optional, Sequence
warnings.filterwarnings("ignore")

def upsample_to_monthly(
    df: pd.DataFrame,
    date_col: str = "date",
    date_index: bool = True,
    groupby_cols: Optional[Sequence[str]] = None,
    interpolate: bool = True,
    mtd_inp: str = None,
) -> pd.DataFrame:

    if date_index:
        df = df.reset_index()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    if groupby_cols is not None:
        grouped = df.groupby(list(groupby_cols))
    else:
        groupby_cols = ['research_submkt_id']
        grouped = df.groupby(list(groupby_cols))

    df_monthly = pd.DataFrame()
    for name, group in grouped:
      if interpolate:
          if mtd_inp is None:
              mtd_inp = 'linear'
          else:
              group_resampled = group.resample('MS', origin='start').interpolate(method=mtd_inp)
              group_resampled['research_submkt_id'] = name
              df_monthly = pd.concat([df_monthly, group_resampled])
      else:
          group_resampled = group.resample('MS', origin='start').ffill(limit=2)
          group_resampled['research_submkt_id'] = name
          df_monthly = pd.concat([df_monthly, group_resampled])

      if date_index:
          df_monthly = df_monthly.set_index(date_col)
      df_monthly.to_csv('/home/zqiao/data_flake/submkt_popstats_monthly.csv')
      return df_monthly

if __name__ == "__main__":
    df = pd.read_csv('/home/zqiao/data_flake/submkt_popstats.csv', index_col=0)
    df_monthly = upsample_to_monthly(df, date_col='date', date_index=False,
                                     groupby_cols=['research_submkt_id'],
                                     interpolate=True,
                                     mtd_inp='polynomial')
    print(df_monthly.head())
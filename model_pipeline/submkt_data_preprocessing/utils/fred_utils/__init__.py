import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
from typing import Optional, Sequence, List
from fredapi import Fred

def get_national_consumption_vars():

    var_mapping = {
        "RETAILIRSA": "inv_to_sales_retail",
        "PCEDG": "pce_dg",
        "PCEND": "pce_ndg",
        "PCEPILFE": "core_pce",
        "RSXFS": "advance_retail_sales",
        "ISRATIO": "inv_to_sales",
        "MRTSIR4423XUSS": "inv_to_sales_furniture",
        "WHLSLRIMSA": "wholesale_inventories",
        "A333RX1Q020SBEA": "imports_durable_goods",
    }

    consumption_vars = {
        f"{k}": v for k, v in var_mapping.items()
    }

    var_li = list(consumption_vars.keys())
    start = datetime.datetime(1992, 2, 1)
    df = pdr.data.DataReader(var_li, "fred", api_key="f5fc7e2012d525af5656009424795011", start=start)
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])

    return df


def adjust_for_inflation(
        df: pd.DataFrame,
        cols: Optional[Sequence[str]] = None,
        groupby_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    fpi = Fred(api_key="f5fc7e2012d525af5656009424795011")
    cpi_series_id = "CPIAUCSL"
    cpi_df = fpi.get_series(series_id=cpi_series_id)
    cpi_df = cpi_df.to_frame(name="cpi_index").reset_index()
    cpi_df.columns = ["date", "cpi_index"]

    def adjust_for_inflation(group_df, cols):
        for col in cols:
            group_df[col] = (group_df[col] * group_df["cpi_index"] / group_df["cpi_index"].iloc[0])
        return group_df

    df = df.merge(cpi_df, how='left', on='date')
    df = df.groupby(groupby_cols, as_index=False).apply(adjust_for_inflation, cols=cols)

    return df




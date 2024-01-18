import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence


def submkt_var_compare_plot(df,
                            groupby_col: Optional[Sequence[str]] = None,
                            x_label: str = 'date',
                            y_label: str = None,
                            legend: bool = True):
    if groupby_col is not None:
        grouped = df.groupby(list(groupby_col))
    else:
        groupby_col = ['research_submkt_id']
        grouped = df.groupby(list(groupby_col))

    fig, ax = plt.subplots()

    if y_label is None:
        y_label = 'real_hedonic_rent_submarket'

    for group_name, group_data in grouped:
        ax.plot(group_data[x_label], group_data[y_label], label=group_name)

    if legend:
        ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Plot for each submkt - {}'.format(y_label))

    # Show the plot
    return plt.show()


def cbre_link_comparison_sf_ratio_plot(df):
    grouped_data_pho = df.groupby('research_submkt_id')
    fig, ax = plt.subplots()
    for group_name, group_data in grouped_data_pho:
        ax.scatter(group_data['submkt_mkt_sf_ratio_link'], group_data['submkt_mkt_sf_ratio_cbre'], label=group_name)

    # Add labels and a legend
    ax.set_xlabel('submkt_mkt_sf_ratio_link')
    ax.set_ylabel('submkt_mkt_sf_ratio_cbre')
    ax.legend(bbox_to_anchor=(1.35, 1), loc='upper right')

    # Display the plot
    return plt.show()


if __name__ == "__main__":
    df = pd.read_csv()
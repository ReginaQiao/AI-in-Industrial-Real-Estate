import multiprocessing as mp
from datetime import datetime as dtm
import numpy as np
import pandas as pd
from darts.models.forecasting.varima import VARIMA
from darts.timeseries import TimeSeries as TS
from sklearn.model_selection import ParameterGrid as PG


def run_varima_pipeline(df):

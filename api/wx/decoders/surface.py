import logging
import os
import time

import numpy as np
import pandas as pd
import pytz
from celery import shared_task

from wx.decoders.insert_raw_data import insert
from wx.decoders.insert_hf_data import insert as insert_hf
from wx.models import Station
from wx.models import Variable
from wx.utils import update_station_variables

logger = logging.getLogger('decoders.surface')
db_logger = logging.getLogger('db')


def parse_first_line_header(file_name):
    """Parse the first line of the header and extract station code
        eg: surface_9958303_2019-05.csv
            surface_9958303.csv
    """
    # get the file name, path may contain _
    file = os.path.basename(file_name)
    code = file.split("_")[1]
    if "." in code:
        code = code.split(".")[0]

    return code


def naive_to_aware(dt, fixed_tz):
    if dt.tzinfo is None and fixed_tz is not None:
        return fixed_tz.localize(dt)
    else:
        return dt


@shared_task
def read_file(filename, highfrequency_data=False, station_object=None, utc_offset=-360, override_data_on_conflict=False):
    """Read a Surface file and return a seq of records or nil in case of error"""

    logger.info('processing %s' % filename)

    start = time.time()

    total_reads = 0

    if station_object is None:
        station_code = parse_first_line_header(filename)
        station_object = Station.objects.get(code=station_code)

    df = pd.read_csv(filename, sep=";", parse_dates=['datetime'])

    df.sort_values(by=['datetime'], inplace=True)

    # remove datetime column from columns
    variable_symbols = df.columns[1:]

    if utc_offset is None:
        fixed_tz = None
    else:
        fixed_tz = pytz.FixedOffset(utc_offset)

    # set timezone if datetime is aware, otherwise use the given timezone
    df['datetime'] = df['datetime'].apply(lambda dt: naive_to_aware(dt, fixed_tz))

    station_id = station_object.id
    for var_symbol in variable_symbols:

        reads = []

        try:
            variable = Variable.objects.get(symbol=var_symbol)
            variable_id = variable.id
        except KeyError as ke:
            print(f"Error variable with symbol {var_symbol} not found!")
            continue

        df_symbol = df[['datetime', var_symbol]].copy()

        # compute time diff in seconds and store that in interval
        df_symbol['datetime_1'] = df_symbol.datetime.shift(1)
        df_symbol['datetime_diff'] = (df_symbol.datetime - df_symbol.datetime_1).dt.total_seconds().replace(np.nan, 0).astype("int32")
        interval_counts = df_symbol['datetime_diff'].value_counts()
        interval = interval_counts.first_valid_index()

        print(f"station={station_object.id}/{station_object.code} "
              f"variable={variable_id}/{var_symbol} "
              f"interval={interval}")

        # remove rows where var_symbol contains NaN
        df_symbol.dropna(subset=[var_symbol], inplace=True)

        # if file contain data for a variable, them add that to update StationVariable
        if df_symbol.shape[0] != 0:
            in_file_station_variables = {variable_id}
            update_station_variables(station_object, in_file_station_variables)

        for idx, row in df_symbol.iterrows():
            columns = [
                station_id,            # station
                variable_id,           # variable
                interval,              # interval seconds
                row["datetime"],       # datetime
                row[var_symbol],       # value
                None,                  # "quality_flag"
                None,                  # "qc_range_quality_flag"
                None,                  # "qc_range_description"
                None,                  # "qc_step_quality_flag"
                None,                  # "qc_step_description"
                None,                  # "qc_persist_quality_flag"
                None,                  # "qc_persist_description"
                None,                  # "manual_flag"
                None,                  # "consisted"
                False                  # "is_daily"
            ]
            reads.append(columns)

        if highfrequency_data:
            insert_hf(reads, override_data_on_conflict)
        else:
            insert(reads, override_data_on_conflict)

        total_reads = total_reads + len(reads)

    end = time.time()

    logger.info(f'Processing file {filename} in {end - start} seconds, '
                f'returning #reads={total_reads}.')





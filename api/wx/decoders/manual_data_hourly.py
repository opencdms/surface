import datetime
import logging
import time

import numpy as np
import pandas as pd
import pytz
from celery import shared_task
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned

from tempestas_api import settings
from wx.decoders.insert_raw_data import insert
from wx.models import Station

logger = logging.getLogger('surface.manual_data')
db_logger = logging.getLogger('db')

cld_variables = {
    "cld1": (1009, 1010, 1011),
    "cld2": (1012, 1013, 1014),
    "cld3": (1015, 1016, 1017),
    "cld4": (1018, 1019, 1020)
}


def get_regular_value(variable_id, value, column_name):
    return ((variable_id, value),)


def get_cld_values(variable_id, value, column_name):
    if value == settings.MISSING_VALUE:
        n_value = settings.MISSING_VALUE
        c_value = settings.MISSING_VALUE
        hh_value = settings.MISSING_VALUE
    else:
        str_value = f"{value:05}"
        n_value = int(str_value[1])
        c_value = int(str_value[2])
        hh_value = int(str_value[3:5])

    variable_ids = cld_variables[column_name]

    n_tuple = (variable_ids[0], n_value)
    c_tuple = (variable_ids[1], c_value)
    hh_tuple = (variable_ids[2], hh_value)

    return (n_tuple, c_tuple, hh_tuple)


def get_w_values(variable_id, value, column_nam):
    if value == settings.MISSING_VALUE:
        w1_value = settings.MISSING_VALUE
        w2_value = settings.MISSING_VALUE
    else:
        str_value = f"{value:02}"
        w1_value = int(str_value[0])
        w2_value = int(str_value[1])

    w1_tuple = (1003, w1_value)
    w2_tuple = (1004, w2_value)

    return (w1_tuple, w2_tuple)


column_names = [
    'day',
    'station_id',
    'hour',
    'visby',
    'cldtot',
    'wnddir',
    'wndspd',
    'tempdb',
    'tempdp',
    'tempwb',
    'vappsr',
    'relhum',
    'pressl',
    'prswx',
    'pastww',
    'nhamt',
    'cldcl',
    'cldcm',
    'cldch',
    'cld1',
    'cld2',
    'cld3',
    'cld4',
    'cldht1'
]

# verificar
variable_dict = {
    'visby': {
        'variable_id': 1000,
        'function': get_regular_value
    },
    'cldtot': {
        'variable_id': 1001,
        'function': get_regular_value
    },
    'wnddir': {
        'variable_id': 55,
        'function': get_regular_value
    },
    'wndspd': {
        'variable_id': 50,
        'function': get_regular_value
    },
    'tempdb': {
        'variable_id': 10,
        'function': get_regular_value
    },
    'tempdp': {
        'variable_id': 19,
        'function': get_regular_value
    },
    'tempwb': {
        'variable_id': 18,
        'function': get_regular_value
    },
    'relhum': {
        'variable_id': 30,
        'function': get_regular_value
    },
    #  'vappsr': ,
    'pressl': {
        'variable_id': 61,
        'function': get_regular_value
    },
    'prswx': {
        'variable_id': 1002,
        'function': get_regular_value
    },
    'pastww': {
        'variable_id': None,
        'function': get_w_values
    },
    'nhamt': {
        'variable_id': 4005,
        'function': get_regular_value
    },
    'cldcl': {
        'variable_id': 1006,
        'function': get_regular_value
    },
    'cldcm': {
        'variable_id': 1007,
        'function': get_regular_value
    },
    'cldch': {
        'variable_id': 1008,
        'function': get_regular_value
    },
    'cld1': {
        'variable_id': None,
        'function': get_cld_values
    },
    'cld2': {
        'variable_id': None,
        'function': get_cld_values
    },
    'cld3': {
        'variable_id': None,
        'function': get_cld_values,

    },
    'cld4': {
        'variable_id': None,
        'function': get_cld_values
    },
    #    'cldht1'
}


def parse_date(day, hour, utc_offset):
    datetime_offset = pytz.FixedOffset(utc_offset)
    date = datetime.datetime.strptime(f'{day} {hour}', '%Y-%m-%d %H')

    return datetime_offset.localize(date)


def parse_line(line, station_id, utc_offset):
    """Parse a manual data row"""

    records_list = []
    day = line['day']
    hour = line['hour']
    parsed_date = parse_date(day, hour, utc_offset)
    seconds = 3600

    for variable_name in variable_dict.keys():
        measurement = line[variable_name]
        if measurement is None or type(measurement) == str:
            measurement = settings.MISSING_VALUE

        variable = variable_dict[variable_name]
        variable_function = variable['function']
        variable_id = variable['variable_id']

        variable_records = variable_function(variable_id, measurement, variable_name)
        for record in variable_records:
            calculed_variable_id = record[0]
            value = record[1]
            records_list.append((station_id, calculed_variable_id, seconds, parsed_date, value, None, None, None, None,
                                 None, None, None, None, None, False))

    return records_list


@shared_task
def read_file(filename, station_object=None, utc_offset=-360, override_data_on_conflict=False):
    """Read a manual data file and return a seq of records or nil in case of error"""

    logger.info('processing %s' % filename)

    start = time.time()
    reads = []
    empty_array = np.array('')
    try:
        source = pd.ExcelFile(filename)

        # iterate over the sheets
        for sheet_name in source.sheet_names:
            sheet_raw_data = source.parse(sheet_name, na_filter=False, names=column_names)
            if not sheet_raw_data.empty:
                sheet_data = sheet_raw_data[pd.to_numeric(sheet_raw_data['hour'], errors='coerce').notnull()]
                station_code = sheet_data[0:1]['station_id'].item()

                try:
                    station_id = Station.objects.get(code=station_code).id
                except (ObjectDoesNotExist, MultipleObjectsReturned) as e:
                    raise Exception(f"Failed to find station by code '{station_code}'. {repr(e)}")

                # filter the sheet day
                sheet_data = sheet_data[pd.to_numeric(sheet_data['hour'], errors='coerce').notnull()]
                for index, row in sheet_data.iterrows():
                    # check if line is not blank
                    if not (row.loc['visby':].squeeze().unique() == empty_array).all():
                        for line_data in parse_line(row, station_id, utc_offset):
                            reads.append(line_data)


    except FileNotFoundError as fnf:
        logger.error(repr(fnf))
        print('No such file or directory {}.'.format(filename))
        raise
    except Exception as e:
        logger.error(repr(e))
        raise

    end = time.time()

    # print(reads)
    insert(reads, override_data_on_conflict)

    logger.info(f'Processing file {filename} in {end - start} seconds, '
                f'returning #reads={len(reads)}.')

    return reads

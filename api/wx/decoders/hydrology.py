import logging
import time
from csv import reader as csv_reader
from datetime import datetime

import psycopg2
import pytz
from celery import shared_task
from psycopg2.extras import execute_values

from tempestas_api import settings
from wx.models import VariableFormat, StationVariable
from wx.utils import check_range_qc_result, update_station_variables

logger = logging.getLogger('surface.hydrology')


def define_date_format(text):
    try:
        datetime.strptime(text[:10], '%Y/%m/%d').astimezone(pytz.UTC)
        return '%Y/%m/%d'
    except ValueError:
        pass

    try:
        datetime.strptime(text[:10], '%Y/%d/%m').astimezone(pytz.UTC)
        return '%Y/%d/%m'
    except ValueError:
        pass

    try:
        datetime.strptime(text[:10], '%m/%d/%Y').astimezone(pytz.UTC)
        return '%m/%d/%Y'
    except ValueError:
        pass

    try:
        datetime.strptime(text[:10], '%d/%m/%Y').astimezone(pytz.UTC)
        return '%d/%m/%Y'
    except ValueError:
        pass


def convert_string_2_datetime(text, utc_offset, cache={}):
    # This 'if' part of the code will be executed only once during the processing of the file.
    if not 'date_format' in cache.keys():
        cache['date_format'] = define_date_format(text)

    datetime_offset = pytz.FixedOffset(utc_offset)
    if (len(text) > 10):
        return datetime.strptime(text, cache['date_format'] + ' %H:%M').astimezone(datetime_offset)
    return datetime.strptime(text, cache['date_format']).astimezone(datetime_offset)


def get_current_station_variable(station_variables_dict, variable_id):
    try:
        return station_variables_dict[variable_id]
    except Exception:
        return None


def parse_line(line, station, lookup_table, station_variables_dict, utc_offset):
    """Remove quotes and returns the line"""
    date_info = convert_string_2_datetime(line[0], utc_offset)

    line_data = []

    for index in range(len(lookup_table)):

        if lookup_table[index] is not None and line[index] != 'NAN' and line[index] != '':
            range_test = check_range_qc_result(
                get_current_station_variable(station_variables_dict, lookup_table[index]), float(line[index]))

            line_data.append(
                (station.id, lookup_table[index], date_info, float(line[index]), range_test['quality_flag'],
                 range_test['result'], range_test['description'], None, None, None, None, None, None))

    return line_data


def parse_first_line_header(line):
    """Parse the first line of the header and extract station code"""

    station_code_name = line[1]

    station_code = station_code_name.split('_')[0]

    return station_code


def parse_second_line_header(station, line):
    """Parse the second line of the header and extract the column names"""

    column_names = line

    variable_format_list = VariableFormat.objects.all()

    lookup_table = {}
    in_file_station_variables = set()

    for variable_format in variable_format_list:
        # print('{} {}'.format(variable_format.variable.id, variable_format.lookup_key))

        lookup_table[variable_format.lookup_key] = variable_format.variable.id

    result = {}

    for index, column_name in enumerate(column_names):

        if column_name in lookup_table.keys():

            variable = lookup_table[column_name]
            in_file_station_variables.add(variable)
        else:

            variable = None

        result[index] = variable

    update_station_variables(station, in_file_station_variables)
    return result


def read_header(file):
    """Read a TOA5 file and return a map with metadata extracted from the header"""
    pass


@shared_task
def read_file(filename, station_object, utc_offset=-360, override_data_on_conflict=False):
    """Read a HYDROLOGY/GENERIC file and return a seq of records or nil in case of error"""

    start = time.time()

    reads = []

    try:
        with open(filename, 'r', encoding='UTF-8') as source:

            reader = csv_reader(source)

            # station_code = parse_first_line_header(next(reader))
            station = station_object
            lookup_table = parse_second_line_header(station, next(reader))

            station_variables_list = StationVariable.objects.filter(station_id=station.id)
            station_variables_dict = {}

            for station_variable in station_variables_list:
                station_variables_dict[station_variable.variable_id] = station_variable

            for r in reader:
                for line_data in parse_line(r, station, lookup_table, station_variables_dict, utc_offset):
                    reads.append(line_data)

    except FileNotFoundError as fnf:
        logger.error(repr(fnf))
        print('No such file or directory {}.'.format(filename))
        exit(1)

    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:
            execute_values(cursor, """
                INSERT INTO raw_data (station_id, variable_id, datetime, measured, quality_flag, qc_range_quality_flag, qc_range_description, qc_step_quality_flag, qc_step_description, qc_persist_quality_flag, qc_persist_description, manual_flag, consisted)
                VALUES %s
                ON CONFLICT DO NOTHING
            """, reads)

        conn.commit()

    end = time.time()

    print('Processing file {} in {} seconds.'.format(filename, end - start))

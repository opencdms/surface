import logging
import time
from csv import reader as csv_reader
from datetime import datetime

import pytz
from celery import shared_task

from wx.decoders.insert_raw_data import insert
from wx.decoders.insert_hf_data import insert as insert_hf
from wx.models import VariableFormat, Station, StationVariable
from wx.utils import update_station_variables

logger = logging.getLogger('surface.hobo')

IGNORE_DAILY_VARIABLES = ['Rain_mm_TOT_24hr', 'Rainfall']


def convert_string_2_datetime(text, utc_offset):
    datetime_offset = pytz.FixedOffset(utc_offset)
    return datetime.strptime(text, '%m/%d/%y %I:%M:%S %p').replace(tzinfo=datetime_offset)


def get_current_station_variable(station_variables_dict, variable_id):
    try:
        return station_variables_dict[variable_id]
    except KeyError:
        return None


def parse_line(line, station, lookup_table, station_variables_dict, utc_offset):
    """Remove quotes and returns the line"""
    date_info = convert_string_2_datetime(line[1], utc_offset)

    line_data = []

    for index in range(len(lookup_table)):

        if lookup_table[index] is not None and line[index] != 'NAN' and line[index] != '':
            line_data.append((station.id,
                              lookup_table[index]['variable_id'],
                              lookup_table[index]['seconds'],
                              date_info, float(line[index]),
                              None, None, None, None, None, None, None, None, None, False))

    return line_data


def parse_first_line_header(line):
    """Parse the first line of the header and extract station code"""

    station_info = line[0].split(':')
    station_code_name = station_info[1]
    station_code = station_code_name.split('_')[0].strip()

    return station_code


def get_column_names(line):
    column_names = []
    for column in line:
        column_names.append(column.split(',')[0])
    return column_names


def parse_second_line_header(station, line):
    """
    Parse the second line of the header and extract the column names
    """
    column_names = get_column_names(line)

    variable_format_list = VariableFormat.objects.all()

    lookup_table = {}
    in_file_station_variables = set()

    for variable_format in variable_format_list:
        lookup_table[variable_format.lookup_key] = {
            'variable_id': variable_format.variable.id,
            'seconds': variable_format.interval.seconds
        }

    result = {}

    for index, column_name in enumerate(column_names):
        if column_name in IGNORE_DAILY_VARIABLES:
            print(f"ignoring variable daily variable {column_name}, need to store only hourly measurements.")
            variable = None
        elif column_name in lookup_table.keys():
            variable = lookup_table[column_name]
            in_file_station_variables.add(variable['variable_id'])
        else:
            variable = None

        result[index] = variable

    update_station_variables(station, in_file_station_variables)
    return result


def read_header(file):
    """Read a TOA5 file and return a map with metadata extracted from the header"""
    pass


@shared_task
def read_file(filename, highfrequency_data=False, station_object=None, utc_offset=-360, override_data_on_conflict=False):
    """Read a TOA5 file and return a seq of records or nil in case of error"""

    start = time.time()

    reads = []

    try:
        with open(filename, 'r', encoding='ISO-8859-1') as source:
            reader = csv_reader(source)

            if station_object is None:
                station_code = parse_first_line_header(next(reader))
                station = Station.objects.get(code=station_code)
            else:
                next(reader)  # skip station line
                station = station_object
                station_code = station.code

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
        logger.error(f'No such file or directory {filename}.')

    if highfrequency_data:
        insert_hf(reads, override_data_on_conflict)
    else:
        insert(reads, override_data_on_conflict)

    end = time.time()

    logger.info(f'Processing file {filename} in {end - start} seconds, '
                f'returning #reads={len(reads)}.')

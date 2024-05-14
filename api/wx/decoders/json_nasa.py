# tentativo di decoder per json nasa

import json
import logging
import time
from datetime import datetime

import pytz
from celery import shared_task

from wx.decoders.insert_raw_data import insert
from wx.decoders.insert_hf_data import insert as insert_hf
from wx.models import VariableFormat, Station, StationVariable
from wx.utils import update_station_variables

logger = logging.getLogger('surface.json_nasa')
db_logger = logging.getLogger('db')

FORMAT = "JSON_NASA"


def convert_string_2_datetime(text, utc_offset):
    """Conversion of date from string to datetime"""

    year = text[0:4]
    month = text[4:6]
    day = text[6:8]
    hour = 0

    if len(text) == 10:
        hour = text[8:10]

    datetime_offset = pytz.FixedOffset(utc_offset)
    date = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), tzinfo=datetime_offset)

    return date


def parse_line(line, station, lookup_table, utc_offset):
    """Conversion of dictionary data to raw_data format"""

    str_date, value = line
    date_info = convert_string_2_datetime(str_date, utc_offset)

    if lookup_table['seconds'] == 86400:
        parsed_line = (station.id, lookup_table['variable_id'], lookup_table['seconds'], date_info, float(value), 1, None, None, None, None, None, None, None, None, True)
    else:
        parsed_line = (station.id, lookup_table['variable_id'], lookup_table['seconds'], date_info, float(value), 1, None, None, None, None, None, None, None, None, False)

    return parsed_line


def fields_extraction(data,  type):
    """Extract the name of the dataset's variable"""

    if type == "FeatureCollection":
        fields = list(data["parameterInformation"].keys())
    elif type == "Feature":
        fields = list(data["parameters"].keys())

    logger.info(f"Fields names: {str(fields)}")

    return fields


def coordinates_extraction(feature):
    """Extract the coordinates of the station from a feature"""

    coordinates = feature["geometry"]["coordinates"]

    logger.info(f"Coordinates: {coordinates}")

    return coordinates


def extract_lookup(station, fields):
    """New version of parse_second_line_header"""

    variable_format_list = VariableFormat.objects.all()

    lookup_table = {}
    in_file_station_variables = set()

    for variable_format in variable_format_list:

        # check if lookup_key is not duplicated
        if variable_format.lookup_key in lookup_table:
            db_logger.error(f"key {variable_format.lookup_key} is already "
                            f"present in lookup table: {variable_format}")

        lookup_table[variable_format.lookup_key] = {
            'variable_id': variable_format.variable.id,
            'seconds': variable_format.interval.seconds
        }

    result = {}

    for field in fields:
        if field in lookup_table.keys():
            variable = lookup_table[field]
            in_file_station_variables.add(variable['variable_id'])
        else:
            db_logger.warning(f'Variable {field} not found while parsing document from station {station.name}')
            variable = None

        result[field] = variable

    # associate new variable on station
    update_station_variables(station, in_file_station_variables)

    """
    forma output:
    {
        nome_campo_dataset:{
            "variable_id":value_id_database,
            "seconds":value
        },
        ...
    }
    """

    return result


def process_feature(feature, fields, reads,  utc_offset):
    """Outsourcing of feature processing"""

    coordinates = coordinates_extraction(feature=feature)
    station = Station.objects.get(longitude=coordinates[0], latitude=coordinates[1])
    station_code = station.code

    logger.info(f"Station code: {station_code}")

    lookup_table = extract_lookup(station, fields)

    logger.info(f"Lookup table: {str(lookup_table)}")

    for field in fields:
        for key, value in feature["properties"]["parameter"][field].items():
            parsed_line = parse_line((key, value), station, lookup_table[field], utc_offset)
            reads.append(parsed_line)


@shared_task
def read_file(filename, highfrequency_data=False, station_object=None, utc_offset=-360, override_data_on_conflict=False):
    """Read a json file and return a seq of records or nil in case of error"""

    logger.info('processing %s' % filename)
    start = time.time()

    reads = []

    try:
        with open(filename, 'r', encoding='UTF-8') as source:
            
            data = json.load(source)

            type = data["type"]
            fields = fields_extraction(data, type)

            if type == "FeatureCollection":
                
                for feature in data["features"]:
                    process_feature(feature, fields, reads,  utc_offset)

            elif type == "Feature":
                process_feature(data, fields, reads, utc_offset)

    except FileNotFoundError as fnf:
        logger.error(repr(fnf))
        print('No such file or directory {}.'.format(filename))
        raise
    except Exception as e:
        logger.error(repr(e))
        raise

    if highfrequency_data:
        insert_hf(reads, override_data_on_conflict)
    else:
        insert(reads, override_data_on_conflict)

    end = time.time()

    logger.info(f'Processing file {filename} in {end - start} seconds, '
                f'returning #reads={len(reads)}.')

    return reads

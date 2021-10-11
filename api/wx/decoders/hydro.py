import calendar
import datetime
import logging
import time

import pandas as pd
import psycopg2
import pytz
from celery import shared_task

from tempestas_api import settings
from wx.decoders.insert_raw_data import insert
from wx.models import RatingCurve, RatingCurveTable

logger = logging.getLogger('surface.manual_data')
db_logger = logging.getLogger('db')

seconds = 43200
river_level_variable_id = 4015
stream_flow_variable_id = 4016


def parse_date(sheet_date, day, hour, utc_offset):
    datetime_offset = pytz.FixedOffset(utc_offset)
    date = sheet_date.replace(day=day, hour=hour)

    return datetime_offset.localize(date)


def create_raw_data_line(station_id, variable_id, seconds, date, measurement):
    return (
        station_id, variable_id, seconds, date, measurement, 1, None, None, None, None, None, None, None, None, False)


def get_interpolated_value(rating_curve, height):
    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT previous_value.h
                      ,previous_value.q
                      ,next_value.h
                      ,next_value.q
                FROM (SELECT h, q FROM wx_ratingcurvetable WHERE rating_curve_id = %(rating_curve_id)s AND h > %(height)s ORDER BY h      LIMIT 1) as next_value
                    ,(SELECT h, q FROM wx_ratingcurvetable WHERE rating_curve_id = %(rating_curve_id)s AND h < %(height)s ORDER BY h DESC LIMIT 1) as previous_value
                """, {'rating_curve_id': rating_curve.id, 'height': height})

            h1, q1, h2, q2 = cursor.fetchone()
            return ((q2 - q1) / (h2 - h1)) * (height - h1) + q1


def get_stream_flow_value(station_id, sheet_date, height):
    if height != settings.MISSING_VALUE:
        rating_curve = RatingCurve.objects.filter(station_id=station_id, start_date__lt=sheet_date).order_by(
            '-start_date').first()
        try:
            stream_flow_measurement = RatingCurveTable.objects.get(rating_curve=rating_curve, h=height).q
        except RatingCurveTable.DoesNotExist:
            try:
                stream_flow_measurement = get_interpolated_value(rating_curve, height)
            except Exception:
                stream_flow_measurement = settings.MISSING_VALUE
    else:
        stream_flow_measurement = settings.MISSING_VALUE

    return create_raw_data_line(station_id, stream_flow_variable_id, seconds, sheet_date, stream_flow_measurement)


def parse_column(station_id, date, parsed_data_list, measurement):
    if measurement is not None and type(measurement) != str:
        hydro_data = create_raw_data_line(station_id, river_level_variable_id, seconds, date, measurement)
        parsed_data_list.append(hydro_data)

        stream_flow_data = get_stream_flow_value(station_id, date, measurement)
        parsed_data_list.append(stream_flow_data)


def parse_line(line, station_id, sheet_date, utc_offset):
    """Remove quotes and returns the line"""

    day = line['DAY']

    parsed_data_list = []

    date_6am = parse_date(sheet_date, day, 6, utc_offset)
    parse_column(station_id, date_6am, parsed_data_list, line['6AM'])

    date_6pm = parse_date(sheet_date, day, 18, utc_offset)
    parse_column(station_id, date_6pm, parsed_data_list, line['6PM'])

    return parsed_data_list


def parse_sheet_date(sheet_name, sheet_raw_data):
    try:
        return datetime.datetime.strptime(sheet_name.replace(' ', '').strip(), '%b%Y')
    except ValueError:
        sheet_month = sheet_raw_data['MAX'][2:3].item().strip()
        sheet_year = sheet_raw_data['MIN'][2:3].item()
        return datetime.datetime.strptime(f"{sheet_month} {sheet_year}".strip(), '%B %Y')


@shared_task
def read_file(filename, station_object, utc_offset=-360, override_data_on_conflict=False):
    """Read a hydro data file and return a seq of records or nil in case of error"""

    logger.info(f'processing {filename}')
    start = time.time()
    station_id = station_object.id

    try:
        reads = []
        source = pd.ExcelFile(filename)

        for sheet_name in source.sheet_names:
            try:
                sheet_raw_data = source.parse(sheet_name, names=['DAY', '6AM', '6PM', 'MEAN', 'MAX', 'MIN'],
                                              na_filter=False, usecols='A:F')
                sheet_date = parse_sheet_date(sheet_name, sheet_raw_data)
            except Exception:
                continue

            first_month_day, last_month_day = calendar.monthrange(sheet_date.year, sheet_date.month)

            sheet_data = sheet_raw_data[pd.to_numeric(sheet_raw_data['DAY'], errors='coerce').notnull()]
            for index, row in sheet_data[sheet_data['DAY'] <= last_month_day].iterrows():
                for data in parse_line(row, station_id, sheet_date, utc_offset):
                    reads.append(data)

    except FileNotFoundError as fnf:
        logger.error(repr(fnf))
        print('No such file or directory {}.'.format(filename))
        raise
    except Exception as e:
        logger.error(repr(e))
        raise

    end = time.time()

    insert(reads, override_data_on_conflict)

    logger.info(f'Processing file {filename} in {end - start} seconds, '
                f'returning #reads={len(reads)}.')

    return reads

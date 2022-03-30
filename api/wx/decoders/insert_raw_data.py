import logging

import numpy as np
import pandas as pd
import psycopg2
import pytz
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone
from psycopg2.extras import execute_values
from tempestas_api import settings
from wx.enums import QualityFlagEnum
from wx.models import QcRangeThreshold, QcStepThreshold, StationVariable, Station, Variable

logger = logging.getLogger('surface')

columns = ["station_id", "variable_id", "seconds", "datetime", "measured", "quality_flag", "qc_range_quality_flag",
           "qc_range_description", "qc_step_quality_flag", "qc_step_description", "qc_persist_quality_flag",
           "qc_persist_description", "manual_flag", "consisted", "is_daily"]

insert_columns = ["station_id", "variable_id", "datetime", "measured", "quality_flag", "qc_range_quality_flag",
                  "qc_range_description", "qc_step_quality_flag", "qc_step_description", "qc_persist_quality_flag",
                  "qc_persist_description", "manual_flag", "consisted", "is_daily", "updated_at", "created_at"]

qc_columns = ["qc_step_quality_flag",
              "qc_step_description",
              "qc_range_quality_flag",
              "qc_range_description",
              "quality_flag"]                  

GOOD = QualityFlagEnum.GOOD.id
NOT_CHECKED = QualityFlagEnum.NOT_CHECKED.id
BAD = QualityFlagEnum.BAD.id


######################## Quality Control #######################

def get_qc_step(thresholds, station_id, variable_id, interval):
    try:
        # Trying to set step thresholds using current station
        _step = QcStepThreshold.objects.get(station_id=station_id, variable_id=variable_id, interval=interval)
        thresholds['step_min'], thresholds['step_max'] = _step.step_min, _step.step_max
        thresholds['step_description'] = 'Custom station Threshold'
    except ObjectDoesNotExist:
        try:
            # Trying to set step thresholds using current station with NULL intervall
            _step = QcStepThreshold.objects.get(station_id=station_id, variable_id=variable_id, interval__isnull=True)        
            thresholds['step_min'], thresholds['step_max'] = _step.step_min, _step.step_max
            thresholds['step_description'] = 'Custom station Threshold'
        except ObjectDoesNotExist:
            try:
                # Trying to set step thresholds using referece station
                _station = Station.objects.get(pk=station_id)
                _step = QcStepThreshold.objects.get(station_id=_station.reference_station_id, variable_id=variable_id, interval=interval)
                thresholds['step_min'], thresholds['step_max'] = _step.step_min, _step.step_max
                thresholds['step_description'] = 'Reference station threshold'
            except ObjectDoesNotExist:
                try:
                    # Trying to set step thresholds using referece station with NULL intervall
                    _station = Station.objects.get(pk=station_id)
                    _step = QcStepThreshold.objects.get(station_id=_station.reference_station_id, variable_id=variable_id, interval__isnull=True)
                    thresholds['step_min'], thresholds['step_max'] = _step.step_min, _step.step_max
                    thresholds['step_description'] = 'Reference station threshold'
                except ObjectDoesNotExist:
                    pass

    return thresholds

def get_qc_range(thresholds, station_id, variable_id, interval, month):
    try:
        # Trying to set range thresholds using current station
        _range = QcRangeThreshold.objects.get(station_id=station_id, variable_id=variable_id, interval=interval, month=month)
        thresholds['range_min'], thresholds['range_max'] = _range.range_min, _range.range_max
        thresholds['range_description'] = 'Custom station Threshold'
    except ObjectDoesNotExist:
        try:
            # Trying to set range thresholds using current station with NULL intervall
            _range = QcRangeThreshold.objects.get(station_id=station_id, variable_id=variable_id, interval__isnull=True, month=month)        
            thresholds['range_min'], thresholds['range_max'] = _range.range_min, _range.range_max
            thresholds['range_description'] = 'Custom station threshold'
        except ObjectDoesNotExist:
            try:
                # Trying to set range thresholds using referece station
                _station = Station.objects.get(pk=station_id)
                _range = QcRangeThreshold.objects.get(station_id=_station.reference_station_id, variable_id=variable_id, interval=interval, month=month)
                thresholds['range_min'], thresholds['range_max'] = _range.range_min, _range.range_max
                thresholds['range_description'] = 'Reference station threshold'
            except ObjectDoesNotExist:
                try:
                    # Trying to set range thresholds using referece station with NULL intervall
                    _station = Station.objects.get(pk=station_id)
                    _range = QcRangeThreshold.objects.get(station_id=_station.reference_station_id, variable_id=variable_id, interval__isnull=True, month=month)
                    thresholds['range_min'], thresholds['range_max'] = _range.range_min, _range.range_max
                    thresholds['range_description'] = 'Reference station threshold'
                except ObjectDoesNotExist:
                    try:
                        # Trying to set range thresholds using global ranges
                        _range = Variable.objects.get(pk=variable_id)                
                        thresholds['range_min'], thresholds['range_max'] = _range.range_min, _range.range_max
                        thresholds['range_description'] = 'Global threshold'
                    except ObjectDoesNotExist:
                        pass;
    return thresholds

def qc_step(seconds, diff_value, diff_datetime, thresholds):
    if 'step_min' not in thresholds or 'step_max' not in thresholds:
        return NOT_CHECKED, "Threshold not found"

    s_min = thresholds['step_min']
    s_max = thresholds['step_max']
    s_des = thresholds['step_description']

    # interval is different
    if seconds != diff_datetime:
        return NOT_CHECKED, "Consecutive value not present"
    elif s_min <= diff_value <= s_max:
        return GOOD, s_des
    elif diff_value < s_min:
        return BAD, s_des
    else:
        return BAD, s_des

def qc_range(value, thresholds):
    if 'range_min' not in thresholds or 'range_max' not in thresholds:
        return NOT_CHECKED, "Threshold not found"

    r_min = thresholds['range_min']
    r_max = thresholds['range_max']
    r_des = thresholds['range_description']

    if r_min <= value <= r_max:
        return GOOD, r_des
    elif value < r_min:
        return BAD, r_des
    else:
        return BAD, r_des

def qc_final(result_step, result_range):
    if BAD in (result_range, result_step):
        return BAD
    elif result_step == NOT_CHECKED:
        return result_range
    elif result_range == NOT_CHECKED:
        return result_step
    else:
        return GOOD

def qc_thresholds(row, thresholds):
    seconds = row.seconds
    value = row.measured

    diff_value = row.diff_value
    diff_datetime = row.diff_datetime

    result_step, msg_step = qc_step(seconds, diff_value, diff_datetime, thresholds)
    result_range, msg_range = qc_range(value, thresholds)

    result_final = qc_final(result_step, result_range)
    result_array = [result_step, msg_step, result_range, msg_range, result_final]

    return result_array

##########################  Functions ##########################

def get_data(raw_data_list):
    now = timezone.now()

    df = pd.DataFrame(raw_data_list, columns=columns)    

    # Convert dates to date object at time zone utc
    df['created_at'] = now
    df['updated_at'] = now
    df['month'] = pd.to_datetime(df['datetime']).dt.month

    reads = []
    for idx, [station_id, variable_id, seconds, month] in df[['station_id', 'variable_id', 'seconds', 'month']].drop_duplicates().iterrows():

        df1 = df.loc[(df.station_id == station_id) &
                     (df.variable_id == variable_id) &
                     (df.seconds == seconds) &
                     (df.month == month)].copy()
                     
        df1.sort_values(by="datetime", inplace=True)

        count = len(df1)
        if count == 0:
            logger.debug(
                f"Skipping station_id={station_id}, variable_id={variable_id}, seconds={seconds} found 0 records, skipping it!")
            continue
        else:
            logger.debug(
                f"Processing station_id={station_id}, variable_id={variable_id}, seconds={seconds} #{count} records.")


        # Defining threshholds
        thresholds = {}
        thresholds = get_qc_step(thresholds=thresholds, station_id=station_id, variable_id=variable_id, interval=seconds)
        thresholds = get_qc_range(thresholds=thresholds, station_id=station_id, variable_id=variable_id, interval=seconds, month=month)

        # Sort values by datetime
        df1.sort_values(by='datetime', inplace=True)

        # Calculating step values
        df1['diff_value'] = df1.measured.diff(periods=1)

        # Calculating step time
        df1['diff_datetime'] = df1.datetime.diff(periods=1).dt.total_seconds().replace(np.nan, 0).astype("int32")

        # Apllying  quality control logic and thresholds
        df1[qc_columns] = df1.apply(lambda row: qc_thresholds(row, thresholds), axis=1, result_type="expand")

        # Replace "" empty string to None/Null
        df1["qc_step_description"].replace("", None)
        df1["qc_range_description"].replace("", None)

        reads.extend(df1[insert_columns].values.tolist())
    return reads

def insert_query(reads, override_data_on_conflict):
    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:

            logger.info(f'Inserting into database #{len(reads)} records.')

            if override_data_on_conflict:
                on_conflict_sql = """
                    ON CONFLICT (station_id, variable_id, datetime)
                    DO UPDATE SET
                        datetime = excluded.datetime,
                        measured = excluded.measured,
                        quality_flag = excluded.quality_flag,
                        qc_range_quality_flag = excluded.qc_range_quality_flag,
                        qc_range_description = excluded.qc_range_description,
                        qc_step_quality_flag = excluded.qc_step_quality_flag,
                        qc_step_description = excluded.qc_step_description,
                        qc_persist_quality_flag = excluded.qc_persist_quality_flag,
                        qc_persist_description = excluded.qc_persist_description,
                        manual_flag = null,
                        consisted = null,
                        updated_at = now()
                """
            else:
                on_conflict_sql = " ON CONFLICT DO NOTHING "

            inserted_raw_data = execute_values(cursor, f"""
                INSERT INTO raw_data (
                        station_id, variable_id, datetime, measured, quality_flag,
                        qc_range_quality_flag, qc_range_description,
                        qc_step_quality_flag, qc_step_description,
                        qc_persist_quality_flag, qc_persist_description,
                        manual_flag, consisted, is_daily, updated_at, created_at)
                VALUES %s
                {on_conflict_sql}
                RETURNING station_id, date_trunc('hour', datetime), now(), now(), is_daily
            """, reads, fetch=True)

            if inserted_raw_data:
                distinct_raw_data = set(inserted_raw_data)
                not_daily_raw_data = filter(lambda data: data[4] == False, distinct_raw_data)
                filtered_raw_data = set(map(lambda raw_data: (raw_data[0], raw_data[1], raw_data[2], raw_data[3]), not_daily_raw_data))

                if filtered_raw_data:
                    execute_values(cursor, """
                        INSERT INTO wx_hourlysummarytask (station_id, datetime, updated_at, created_at)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                    """, filtered_raw_data)

                    station_id = reads[0][0]
                    station_fixed_offset = pytz.FixedOffset(Station.objects.get(pk=station_id).utc_offset_minutes)
                    # When a datetime is inserted on the database, the inserted value returns converted to UTC timezone
                    # Convert UTC datetime to the station_fixed_offset and then transform datetime field in date to insert in wx_dailysummarytask table

                    filtered_raw_data = map(lambda raw_data: (raw_data[0], raw_data[1].astimezone(station_fixed_offset).date(), raw_data[2], raw_data[3]),
                                            filtered_raw_data)
                    filtered_raw_data = set(filtered_raw_data)

                    execute_values(cursor, """
                        INSERT INTO wx_dailysummarytask (station_id, date, updated_at, created_at)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                    """, filtered_raw_data)

        conn.commit()

def update_stationvariable(reads):
    # holds last value for (station_id, variable_id) to update StationVariable last_data_datetime
    update_station_variable = {}
    for read in reads:
        station_id = read[0]
        variable_id = read[1]
        observation_datetime = read[2]
        observation_value = read[3]

        if (station_id, variable_id) not in update_station_variable:
            update_station_variable[(station_id, variable_id)] = [
                station_id, variable_id, observation_datetime, observation_value
            ]
        else:
            [prev_station_id, prev_var_id, prev_datetime, prev_value] = update_station_variable[
                (station_id, variable_id)]
            if prev_datetime < observation_datetime:
                update_station_variable[(station_id, variable_id)] = [
                    station_id, variable_id, observation_datetime, observation_value
                ]


    for read in update_station_variable.values():

        station_id = read[0]
        variable_id = read[1]
        observation_datetime = read[2]
        observation_value = read[3]

        station_variable, created = StationVariable.objects.get_or_create(station_id=station_id,
                                                                          variable_id=variable_id)

        if station_variable.last_data_datetime is None or observation_datetime >= station_variable.last_data_datetime:
            station_variable.last_data_datetime = observation_datetime
            station_variable.last_data_value = observation_value
            station_variable.last_data_code = None

            station_variable.save()

            logger.info(f'Updating StationVariable {station_id} {variable_id} '
                        f'{observation_datetime} {observation_value}')

############################# Main #############################

def insert(raw_data_list, override_data_on_conflict=False):
    # Extracting and formating data from raw data list
    reads = get_data(raw_data_list)

    # Inserting new data
    insert_query(reads, override_data_on_conflict)

    # Updating "station varaiable" table
    update_stationvariable(reads)
import datetime
import logging

import numpy as np
import pandas as pd
import psycopg2
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone
from psycopg2.extras import execute_values
from tempestas_api import settings
from wx.enums import QualityFlagEnum
from wx.models import QcRangeThreshold, QcStepThreshold, StationVariable, Station, Variable

logger = logging.getLogger('surface')

columns = ["station_id", "variable_id", "seconds", "datetime", "measured", "quality_flag", "qc_range_quality_flag",
           "qc_range_description", "qc_step_quality_flag", "qc_step_description", "qc_persist_quality_flag",
           "qc_persist_description", "manual_flag", "consisted", "is_daily", "remarks", "observer", "code"]

insert_columns = ["station_id", "variable_id", "datetime", "measured", "quality_flag", "qc_range_quality_flag",
                  "qc_range_description", "qc_step_quality_flag", "qc_step_description", "qc_persist_quality_flag",
                  "qc_persist_description", "manual_flag", "consisted", "is_daily", "remarks", "observer", "code",
                  "updated_at", "created_at"]

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
                    try:
                        # Trying to set range thresholds using global ranges
                        _station = Station.objects.get(pk=station_id)
                        _step = Variable.objects.get(pk=variable_id)
                        if _station.is_automatic:
                            if _step.step_hourly is not None:
                                thresholds['step_min'], thresholds['step_max'] = -_step.step_hourly, _step.step_hourly
                                thresholds['step_description'] = 'Global threshold (Automatic)'
                        else:
                            if _step.step is not None:
                                thresholds['step_min'], thresholds['step_max'] = -_step.step, _step.step
                                thresholds['step_description'] = 'Global threshold (Manual)'
                    except ObjectDoesNotExist:
                        pass;

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
                        _station = Station.objects.get(pk=station_id)
                        _range = Variable.objects.get(pk=variable_id)
                        if _station.is_automatic:
                            thresholds['range_min'], thresholds['range_max'] = _range.range_min_hourly, _range.range_max_hourly
                            thresholds['range_description'] = 'Global threshold (Automatic)'
                        else:
                            thresholds['range_min'], thresholds['range_max'] = _range.range_min, _range.range_max
                            thresholds['range_description'] = 'Global threshold (Manual)'                                       
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

    if s_min <= diff_value <= s_max:
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

    if r_min is None or r_max is None:
        return NOT_CHECKED, "Threshold not found"           

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

    # convert dates to date object at time zone utc
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

        try:
            process_qc = Variable.objects.filter(pk=variable_id, variable_type="Numeric").exists()
        except ObjectDoesNotExist:
            process_qc = False

        if process_qc:
            thresholds = {}
            thresholds = get_qc_step(thresholds=thresholds, station_id=station_id, variable_id=variable_id, interval=seconds)
            thresholds = get_qc_range(thresholds=thresholds, station_id=station_id, variable_id=variable_id, interval=seconds, month=month)

            # Sort values by datetime
            df1.sort_values(by='datetime', inplace=True)

            # Calculating step values
            df1['diff_value'] = df1.measured.diff(periods=1)

            # Calculating step time
            df1['diff_datetime'] = df1.datetime.diff(periods=1).dt.total_seconds().replace(np.nan, 0).astype("int32")

            df1[qc_columns] = df1.apply(lambda row: qc_thresholds(row, thresholds), axis=1, result_type="expand")

            # replace "" empty string to None/null
            df1["qc_step_description"].replace("", None)
            df1["qc_range_description"].replace("", None)

            df1 = df1.assign(code=None)
        else:
            df1 = df1.assign(measured=settings.MISSING_VALUE)

        reads.extend(df1[insert_columns].values.tolist())        
    return reads

def insert_query(reads, station_id, date, override_data_on_conflict):
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
                        remarks = excluded.remarks,
                        observer = excluded.observer,
                        code = excluded.code,
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
                        manual_flag, consisted, is_daily, remarks, observer, code, updated_at, created_at)
                VALUES %s
                {on_conflict_sql}
                RETURNING station_id, variable_id, datetime
            """, reads, fetch=True)

            if inserted_raw_data:
                now = datetime.datetime.now()
                filtered_raw_data = set(map(lambda raw_data: (
                    raw_data[0], raw_data[2].replace(minute=0, second=0, microsecond=0), now, now), inserted_raw_data))

                execute_values(cursor, """
                    INSERT INTO wx_hourlysummarytask (station_id, datetime, created_at, updated_at)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, filtered_raw_data)

                cursor.execute(f"""
                    INSERT INTO wx_dailysummarytask (station_id, date, created_at, updated_at)
                    VALUES ({station_id}, '{date}', now(), now())
                    ON CONFLICT DO NOTHING
                """)

        conn.commit()

def update_stationvariable(reads):
    # holds last value for (station_id, variable_id) to update StationVariable last_data_datetime
    update_station_variable = {}
    for read in reads:
        station_id = read[0]
        variable_id = read[1]
        observation_datetime = read[2]
        observation_value = read[3]
        observation_code = read[16]

        if (station_id, variable_id) not in update_station_variable:
            update_station_variable[(station_id, variable_id)] = [
                station_id, variable_id, observation_datetime, observation_value, observation_code
            ]
        else:
            [prev_station_id, prev_var_id, prev_datetime, prev_value, prev_code] = update_station_variable[
                (station_id, variable_id)]
            if prev_datetime < observation_datetime:
                update_station_variable[(station_id, variable_id)] = [
                    station_id, variable_id, observation_datetime, observation_value, observation_code
                ]

    for read in update_station_variable.values():

        station_id = read[0]
        variable_id = read[1]
        observation_datetime = read[2]
        observation_value = read[3]
        observation_code = read[4]

        station_variable, created = StationVariable.objects.get_or_create(station_id=station_id,
                                                                          variable_id=variable_id)

        if station_variable.last_data_datetime is None or observation_datetime >= station_variable.last_data_datetime:
            station_variable.last_data_datetime = observation_datetime
            station_variable.last_data_value = observation_value
            station_variable.last_data_code = observation_code
            station_variable.save()

            logger.info(f'Updating StationVariable {station_id} {variable_id} '
                        f'{observation_datetime} {observation_value}')

############################# Main #############################

def insert(raw_data_list, date, station_id, override_data_on_conflict=False, utc_offset_minutes=-360):
    # Extracting and formating data from raw data list
    reads = get_data(raw_data_list)

    # Inserting new data
    insert_query(reads, station_id, date, override_data_on_conflict)

    # Updating "station varaiable" table
    update_stationvariable(reads)

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

GOOD = QualityFlagEnum.GOOD.id
NOT_CHECKED = QualityFlagEnum.NOT_CHECKED.id
BAD = QualityFlagEnum.BAD.id

def get_qc_range(thresholds, station_id, variable_id, interval, month):
    # Trying to set range thresholds using current station
    try:
        _range = QcRangeThreshold.objects.get(station_id=station_id, variable_id=variable_id, interval=interval, month=month)        
        thresholds['range_min'] = _range.range_min
        thresholds['range_max'] = _range.range_max
    except ObjectDoesNotExist:
        # Trying to set range thresholds using referece station
        try:
            _station = Station.objects.get(pk=station_id)
            _range = QcRangeThreshold.objects.get(station_id=_station.reference_station_id, variable_id=variable_id, interval=interval, month=month)
            thresholds['range_min'] = _range.range_min
            thresholds['range_max'] = _range.range_max
        except ObjectDoesNotExist:
            # Trying to set range thresholds using global ranges
            try:
                _range = Variable.objects.get(pk=variable_id)                
                thresholds['range_min'] = _range.range_min
                thresholds['range_max'] = _range.range_max
            except ObjectDoesNotExist:
              pass

    return thresholds


def get_qc_step(thresholds, station_id, variable_id, interval, month):
    try:
        _step = QcStepThreshold.objects.get(station_id=station_id, variable_id=variable_id, interval=interval)
        thresholds['step_min'] = _step.step_min
        thresholds['step_max'] = _step.step_max
    except ObjectDoesNotExist:
        pass
    return thresholds


def qc_step(seconds, diff_value, diff_datetime, thresholds):
    if 'step_min' not in thresholds or 'step_min' not in thresholds:
        return NOT_CHECKED, "Threshold not found"

    s_min = thresholds['step_min']
    s_max = thresholds['step_max']

    # interval is different
    if seconds != diff_datetime:
        return NOT_CHECKED, "Consecutive value not present"
    elif s_min <= diff_value <= s_max:
        return GOOD, ""
    elif diff_value < s_min:
        msg = f"{diff_value} < {s_min}"
        return BAD, msg
    else:
        msg = f"{diff_value} > {s_max}"
        return BAD, msg


def qc_range(value, thresholds):
    if 'range_min' not in thresholds or 'range_min' not in thresholds:
        return NOT_CHECKED, "Threshold not found"

    r_min = thresholds['range_min']
    r_max = thresholds['range_max']

    if r_min <= value <= r_max:
        return GOOD, ""
    elif value < r_min:
        return BAD, f"{value} < {r_min}"
    else:
        return BAD, f"{value} > {r_max}"


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


def insert(raw_data_list, override_data_on_conflict=False):
    df = pd.DataFrame(raw_data_list, columns=columns)

    now = timezone.now()
    # convert dates to date object at time zone utc
    df['created_at'] = now
    df['updated_at'] = now
    df['month'] = pd.to_datetime(df['datetime']).dt.month

    reads = []
    for idx, [station_id, variable_id, seconds, month] in df[
        ['station_id', 'variable_id', 'seconds', 'month']].drop_duplicates().iterrows():

        df1 = df.loc[(df.station_id == station_id) & (df.variable_id == variable_id) & (df.seconds == seconds) & (
                df.month == month)].copy()
        df1.sort_values(by="datetime", inplace=True)

        count = len(df1)

        if count == 0:
            logger.debug(
                f"skipping station_id={station_id}, variable_id={variable_id}, seconds={seconds} found 0 records, skipping it!")
            continue
        else:
            logger.debug(
                f"processing station_id={station_id}, variable_id={variable_id}, seconds={seconds} #{count} records.")

        
        # defining threshholds
        thresholds = {}
        thresholds = get_qc_step(thresholds=thresholds, station_id=station_id, variable_id=variable_id, interval=seconds, month=month)
        thresholds = get_qc_range(thresholds=thresholds, station_id=station_id, variable_id=variable_id, interval=seconds, month=month)


        # sort values by datetime
        df1.sort_values(by='datetime', inplace=True)

        df1["measured_1"] = df1.measured.shift(1)
        df1["datetime_1"] = df1.datetime.shift(1)

        df1['diff_value'] = df1.measured - df1.measured_1
        df1['diff_datetime'] = (df1.datetime - df1.datetime_1).dt.total_seconds().replace(np.nan, 0).astype("int32")

        df1[["qc_step_quality_flag", "qc_step_description",
             "qc_range_quality_flag", "qc_range_description",
             "quality_flag"]] = df1.apply(lambda row: qc_thresholds(row, thresholds), axis=1, result_type="expand")

        # replace "" empty string to None/null
        df1.loc[df1["qc_step_description"] == "", "qc_step_description"] = None
        df1.loc[df1["qc_range_description"] == "", "qc_range_description"] = None

        reads.extend(df1[insert_columns].values.tolist())

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
                filtered_raw_data = set(
                    map(lambda raw_data: (raw_data[0], raw_data[1], raw_data[2], raw_data[3]), not_daily_raw_data))

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

                    filtered_raw_data = map(lambda raw_data: (
                        raw_data[0], raw_data[1].astimezone(station_fixed_offset).date(), raw_data[2], raw_data[3]),
                                            filtered_raw_data)
                    filtered_raw_data = set(filtered_raw_data)

                    execute_values(cursor, """
                        INSERT INTO wx_dailysummarytask (station_id, date, updated_at, created_at)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                    """, filtered_raw_data)

        conn.commit()

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

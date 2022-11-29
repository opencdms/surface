import logging, pytz, psycopg2
import numpy as np
import pandas as pd
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone
from psycopg2.extras import execute_values
from tempestas_api import settings
from wx.models import StationVariable

logger = logging.getLogger('surface')

columns = ["station_id", "variable_id", "seconds", "datetime", "measured"]

insert_columns = ["station_id", "variable_id", "datetime", "measured", "updated_at", "created_at"]

##########################  Functions ##########################

def get_data(raw_data_list):
    now = timezone.now()

    df = pd.DataFrame(raw_data_list)    
    df = df.iloc[:,:len(columns)]
    df.columns=columns

    # Convert dates to date object at time zone utc
    df['created_at'] = now
    df['updated_at'] = now

    reads = []
    for idx, [station_id, variable_id, seconds] in df[['station_id', 'variable_id', 'seconds']].drop_duplicates().iterrows():
        df1 = df.loc[(df.station_id == station_id) &
                     (df.variable_id == variable_id) &
                     (df.seconds == seconds)].copy()
                     
        df1.sort_values(by="datetime", inplace=True)

        count = len(df1)
        if count == 0:
            logger.debug(
                f"Skipping station_id={station_id}, variable_id={variable_id}, seconds={seconds} found 0 records, skipping it!")
            continue
        else:
            logger.debug(
                f"Processing station_id={station_id}, variable_id={variable_id}, seconds={seconds} #{count} records.")

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
                        updated_at = now()
                """
            else:
                on_conflict_sql = " ON CONFLICT DO NOTHING "

            inserted_hf_data =  execute_values(cursor, f"""
                INSERT INTO wx_highfrequencydata (
                        station_id, variable_id, datetime, measured, updated_at, created_at)
                VALUES %s
                {on_conflict_sql}
                RETURNING station_id, variable_id, datetime, now(), now()
            """, reads, fetch=True)

            if inserted_hf_data:

                df = pd.DataFrame(inserted_hf_data, columns = ['station_id', 'variable_id', 'datetime', 'created_at','updated_at'])

                filtered_hf_data = []
                for idx, [station_id, variable_id] in df[['station_id', 'variable_id']].drop_duplicates().iterrows():
                    df1 = df.loc[(df.station_id == station_id) & (df.variable_id == variable_id)]

                    created_at = df1.created_at.max()
                    updated_at = df1.updated_at.max()                    

                    entry = [created_at, updated_at, station_id, variable_id, df1.datetime.min(), df1.datetime.max()]

                    filtered_hf_data.append(entry)

                if filtered_hf_data:
                    execute_values(cursor, """
                        INSERT INTO wx_hfsummarytask (created_at, updated_at, station_id, variable_id, start_datetime, end_datetime)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                    """, filtered_hf_data)

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
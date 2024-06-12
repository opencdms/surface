import pandas as pd

import argparse, logging, psycopg2, datetime, os, json

def get_data(initial_datetime, final_datetime, data_source, series, interval):
    DB_NAME=os.getenv('SURFACE_DB_NAME')
    DB_USER=os.getenv('SURFACE_DB_USER')
    DB_PASSWORD=os.getenv('SURFACE_DB_PASSWORD')
    DB_HOST=os.getenv('SURFACE_DB_HOST')

    config = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST}"

    series = [(row['station_id'], row['variable_id']) for row in series]

    if (data_source=='raw_data'):
      dfs = []
      ini_day = initial_datetime;
      while (ini_day <= final_datetime):
        fin_day = ini_day + datetime.timedelta(days=1)
        fin_day = fin_day.replace(hour=0, minute=0, second=0, microsecond=0) 

        fin_day = min(fin_day, final_datetime)

        if fin_day <= final_datetime:
          query = f"""
            WITH time_series AS(
              SELECT 
                timestamp AS datetime
              FROM
                GENERATE_SERIES(
                  '{ini_day}'::TIMESTAMP
                  ,'{fin_day}'::TIMESTAMP
                  ,'{interval} SECONDS'
                ) AS timestamp
              WHERE timestamp BETWEEN '{ini_day}' AND '{fin_day}'
            )          
            ,series AS (
                SELECT station_id, variable_id
                FROM UNNEST(ARRAY{series}) AS t(station_id int, variable_id int)
            )
            ,processed_data AS (
              SELECT datetime
                  ,station_id
                  ,var.id as variable_id
                  ,COALESCE(CASE WHEN var.variable_type ilike 'code' THEN data.code ELSE data.measured::varchar END, '-99.9') AS value
              FROM raw_data data
              LEFT JOIN wx_variable var ON data.variable_id = var.id
              WHERE data.datetime >= '{ini_day}'
                AND data.datetime < '{fin_day}'
                AND (station_id, variable_id) IN (SELECT station_id, variable_id FROM series)
            )
            SELECT 
              ts.datetime AS datetime
              ,series.variable_id AS variable_id
              ,series.station_id AS station_id
              ,COALESCE(data.value, '-99.9') AS value
            FROM time_series ts
            CROSS JOIN series
            LEFT JOIN processed_data AS data
              ON data.datetime = ts.datetime
              AND data.variable_id = series.variable_id
              AND data.station_id = series.station_id;
          """
        else:
          query = f"""
            WITH time_series AS(
              SELECT 
                timestamp AS datetime
              FROM
                GENERATE_SERIES(
                  '{ini_day}'::TIMESTAMP
                  ,'{fin_day}'::TIMESTAMP
                  ,'{interval} SECONDS'
                ) AS timestamp
              WHERE timestamp BETWEEN '{ini_day}' AND '{fin_day}'
            )
            ,series AS (
                SELECT station_id, variable_id
                FROM UNNEST(ARRAY{series}) AS t(station_id int, variable_id int)
            )
            ,processed_data AS (
              SELECT datetime
                  ,station_id
                  ,var.id as variable_id
                  ,COALESCE(CASE WHEN var.variable_type ilike 'code' THEN data.code ELSE data.measured::varchar END, '-99.9') AS value
              FROM raw_data data
              LEFT JOIN wx_variable var ON data.variable_id = var.id
              WHERE data.datetime BETWEEN '{ini_day}' AND '{fin_day}'
                AND (station_id, variable_id) IN (SELECT station_id, variable_id FROM series)
            )
            SELECT 
              ts.datetime AS datetime
              ,series.variable_id AS variable_id
              ,series.station_id AS station_id
              ,COALESCE(data.value, '-99.9') AS value
            FROM time_series ts
            CROSS JOIN series
            LEFT JOIN processed_data AS data
              ON data.datetime = ts.datetime
              AND data.variable_id = series.variable_id
              AND data.station_id = series.station_id;
          """

        with psycopg2.connect(config) as conn:
          with conn.cursor() as cursor:
            logging.info(query)
            cursor.execute(query)
            data = cursor.fetchall()

        dfs.append(pd.DataFrame(data))

        ini_day += datetime.timedelta(days=1)
        ini_day = ini_day.replace(hour=0, minute=0, second=0, microsecond=0)

        if ini_day == final_datetime:
          break

      df = pd.concat(dfs)
      return df
    else:
      if (data_source=='hourly_summary'):
        query = f'''
            WITH time_series AS(
              SELECT 
                timestamp AS datetime
              FROM
                GENERATE_SERIES(
                  DATE_TRUNC('HOUR', '{initial_datetime}'::TIMESTAMP)
                  ,DATE_TRUNC('HOUR', '{final_datetime}'::TIMESTAMP)
                  ,'1 HOUR'
                ) AS timestamp
              WHERE timestamp BETWEEN '{initial_datetime}' AND '{final_datetime}'
            )       
            ,series AS (
                SELECT station_id, variable_id
                FROM UNNEST(ARRAY{series}) AS t(station_id int, variable_id int)
            )
            ,processed_data AS (
              SELECT
                datetime
                ,station_id
                ,var.id as variable_id
                ,COALESCE(CASE 
                  WHEN var.sampling_operation_id in (1,2) THEN data.avg_value::real
                  WHEN var.sampling_operation_id = 3      THEN data.min_value
                  WHEN var.sampling_operation_id = 4      THEN data.max_value
                  WHEN var.sampling_operation_id = 6      THEN data.sum_value
                  ELSE data.sum_value END, '-99.9') as value
              FROM hourly_summary data
              LEFT JOIN wx_variable var ON data.variable_id = var.id
              WHERE data.datetime BETWEEN '{initial_datetime}' AND '{final_datetime}'
                AND (station_id, variable_id) IN (SELECT station_id, variable_id FROM series)
            )
            SELECT 
              ts.datetime AS datetime
              ,series.variable_id AS variable_id
              ,series.station_id AS station_id
              ,COALESCE(data.value, '-99.9') AS value
            FROM time_series ts
            CROSS JOIN series
            LEFT JOIN processed_data AS data
              ON data.datetime = ts.datetime
              AND data.variable_id = series.variable_id
              AND data.station_id = series.station_id;
        '''    
      elif (data_source=='daily_summary'):      
        query = f'''
            WITH time_series AS(
              SELECT 
                timestamp::DATE AS date
              FROM
                GENERATE_SERIES(
                  DATE_TRUNC('DAY', '{initial_datetime}'::TIMESTAMP)
                  ,DATE_TRUNC('DAY', '{final_datetime}'::TIMESTAMP)
                  ,'1 DAY'
                ) AS timestamp
              WHERE timestamp BETWEEN '{initial_datetime}' AND '{final_datetime}'
            )       
            ,series AS (
                SELECT station_id, variable_id
                FROM UNNEST(ARRAY{series}) AS t(station_id int, variable_id int)
            )
            ,processed_data AS (
              SELECT
                day
                ,station_id
                ,var.id as variable_id
                ,COALESCE(CASE 
                  WHEN var.sampling_operation_id in (1,2) THEN data.avg_value::real
                  WHEN var.sampling_operation_id = 3      THEN data.min_value
                  WHEN var.sampling_operation_id = 4      THEN data.max_value
                  WHEN var.sampling_operation_id = 6      THEN data.sum_value
                  ELSE data.sum_value END, '-99.9') as value
              FROM daily_summary data
              LEFT JOIN wx_variable var ON data.variable_id = var.id
              WHERE data.day BETWEEN '{initial_datetime}' AND '{final_datetime}'
                AND (station_id, variable_id) IN (SELECT station_id, variable_id FROM series)
            )
            SELECT 
              ts.date AS date
              ,series.variable_id AS variable_id
              ,series.station_id AS station_id
              ,COALESCE(data.value, '-99.9') AS value
            FROM time_series ts
            CROSS JOIN series
            LEFT JOIN processed_data AS data
              ON data.day = ts.date
              AND data.variable_id = series.variable_id
              AND data.station_id = series.station_id;
        '''
      elif (data_source=='monthly_summary'):
        query = f'''
            WITH time_series AS(
              SELECT 
                timestamp::DATE AS date
              FROM
                GENERATE_SERIES(
                  DATE_TRUNC('MONTH', '{initial_datetime}'::TIMESTAMP)
                  ,DATE_TRUNC('MONTH', '{final_datetime}'::TIMESTAMP)
                  ,'1 MONTH'
                ) AS timestamp
              WHERE timestamp BETWEEN '{initial_datetime}' AND '{final_datetime}'
            )       
            ,series AS (
                SELECT station_id, variable_id
                FROM UNNEST(ARRAY{series}) AS t(station_id int, variable_id int)
            )
            ,processed_data AS (
              SELECT
                date
                ,station_id
                ,var.id as variable_id
                ,COALESCE(CASE 
                  WHEN var.sampling_operation_id in (1,2) THEN data.avg_value::real
                  WHEN var.sampling_operation_id = 3      THEN data.min_value
                  WHEN var.sampling_operation_id = 4      THEN data.max_value
                  WHEN var.sampling_operation_id = 6      THEN data.sum_value
                  ELSE data.sum_value END, '-99.9') as value
              FROM monthly_summary data
              LEFT JOIN wx_variable var ON data.variable_id = var.id
              WHERE data.date BETWEEN '{initial_datetime}' AND '{final_datetime}'
                AND (station_id, variable_id) IN (SELECT station_id, variable_id FROM series)
            )
            SELECT 
              ts.date AS date
              ,series.variable_id AS variable_id
              ,series.station_id AS station_id
              ,COALESCE(data.value, '-99.9') AS value
            FROM time_series ts
            CROSS JOIN series
            LEFT JOIN processed_data AS data
              ON data.date = ts.date
              AND data.variable_id = series.variable_id
              AND data.station_id = series.station_id;        
        '''
      elif (data_source=='yearly_summary'):
        query = f'''
            WITH time_series AS(
              SELECT 
                timestamp::DATE AS date
              FROM
                GENERATE_SERIES(
                  DATE_TRUNC('YEAR', '{initial_datetime}'::TIMESTAMP)
                  ,DATE_TRUNC('YEAR', '{final_datetime}'::TIMESTAMP)
                  ,'1 YEAR'
                ) AS timestamp
              WHERE timestamp BETWEEN '{initial_datetime}' AND '{final_datetime}'
            )       
            ,series AS (
                SELECT station_id, variable_id
                FROM UNNEST(ARRAY{series}) AS t(station_id int, variable_id int)
            )
            ,processed_data AS (
              SELECT
                date
                ,station_id
                ,var.id as variable_id
                ,COALESCE(CASE 
                  WHEN var.sampling_operation_id in (1,2) THEN data.avg_value::real
                  WHEN var.sampling_operation_id = 3      THEN data.min_value
                  WHEN var.sampling_operation_id = 4      THEN data.max_value
                  WHEN var.sampling_operation_id = 6      THEN data.sum_value
                  ELSE data.sum_value END, '-99.9') as value
              FROM yearly_summary data
              LEFT JOIN wx_variable var ON data.variable_id = var.id
              WHERE data.date BETWEEN '{initial_datetime}' AND '{final_datetime}'
                AND (station_id, variable_id) IN (SELECT station_id, variable_id FROM series)
            )
            SELECT 
              ts.date AS date
              ,series.variable_id AS variable_id
              ,series.station_id AS station_id
              ,COALESCE(data.value, '-99.9') AS value
            FROM time_series ts
            CROSS JOIN series
            LEFT JOIN processed_data AS data
              ON data.date = ts.date
              AND data.variable_id = series.variable_id
              AND data.station_id = series.station_id;        
        '''               

      with psycopg2.connect(config) as conn:
        with conn.cursor() as cursor:
          logging.info(query)
          cursor.execute(query)
          data = cursor.fetchall()

      df = pd.DataFrame(data)
    return df


def main(data_source, file_format, interval, initial_date, initial_time, final_date, final_time, series):
    series = json.loads(series)

    # Set up logging
    logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    # Your code here
    # logging.info(f"data_source: {data_source}")
    # logging.info(f"file_format: {file_format}")
    # logging.info(f"interval: {interval}")
    # logging.info(f"initial_date: {initial_date}")
    # logging.info(f"initial_time: {initial_time}")
    # logging.info(f"final_date: {final_date}")
    # logging.info(f"final_time: {final_time}")
    # logging.info(f"series: {series}")


    initial_datetime = datetime.datetime.strptime(initial_date+' '+initial_time, '%Y-%m-%d %H:%M')
    final_datetime = datetime.datetime.strptime(final_date+' '+final_time, '%Y-%m-%d %H:%M')

    df = get_data(initial_datetime, final_datetime, data_source, series, interval)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_source', type=str, required=True, help='Data source')
    parser.add_argument('--file_format', type=str, required=True, help='File format')
    parser.add_argument('--interval', type=str, required=True, help='Interval')
    parser.add_argument('--initial_date', type=str, required=True, help='Initial date')
    parser.add_argument('--initial_time', type=str, required=True, help='Initial time')
    parser.add_argument('--final_date', type=str, required=True, help='Final date')
    parser.add_argument('--final_time', type=str, required=True, help='Final time')
    parser.add_argument('--series', type=str, required=True, help='Series')

    args = parser.parse_args()
    
    df = main(args.data_source, args.file_format, args.interval, args.initial_date, args.initial_time, args.final_date, args.final_time, args.series)

    # data = {
    #     'Name': ['Alice', 'Bob', 'Charlie'],
    #     'Age': [25, 30, 22],
    #     'City': ['Seattle', 'New York', 'Los Angeles']
    # }

    # df = pd.DataFrame(data)

    print(df.to_csv(index=False))



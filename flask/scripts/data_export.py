from flask import Flask, Response, request, jsonify
import psycopg2
import pandas as pd
import logging
from flask_restful import Api, Resource, reqparse
import datetime
import io
from flask_cors import CORS
import os


app = Flask(__name__)
CORS(app)

api = Api(app)

def get_user_id(token):
  DB_NAME=os.getenv('SURFACE_DB_NAME')
  DB_USER=os.getenv('SURFACE_DB_USER')
  DB_PASSWORD=os.getenv('SURFACE_DB_PASSWORD')
  DB_HOST=os.getenv('SURFACE_DB_HOST')

  config = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST}"

  with psycopg2.connect(config) as conn:
    with conn.cursor() as cursor:
      query = f"""
        SELECT user_id
        FROM authtoken_token
        WHERE key = '{token}'
      """
      logging.info(query)        
      cursor.execute(query)
      data = cursor.fetchall()
  logging.info(data)
  logging.info(len(data))

  if len(data) > 0:
    if len(data[0]) > 0:
      user_id = data[0][0]
      return user_id
    return None
  return None

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

class DataExport(Resource):
    def post(self):
      logging.basicConfig(level=logging.INFO)

      # User Authentication
      auth_header = request.headers.get('Authorization')
      content_type_header = request.headers.get('Content-Type')

      token = auth_header.replace('Token ', '')

      user_id = get_user_id(token)
      is_authenticated = user_id is None
      if is_authenticated:
        response = jsonify({'message': 'Authentication failed'})
        response.status_code = 401  # Unauthorized
        return response

      #     initial_date = '2021-01-01'
      #     initial_time = '00:00'
      #     final_date = '2021-02-01'
      #     final_time = '00:00'
      #     series = [{'station_id': 193, 'variable_id': 0}]

      #     data_source = 'raw_data'

      data_source = request.args.get('data_source')
      file_format = request.args.get('file_format')
      interval = request.args.get('interval')

      initial_date = request.args.get('initial_date')
      initial_time = request.args.get('initial_time')
      final_date = request.args.get('final_date')
      final_time = request.args.get('final_time')

      series = request.get_json()

      initial_datetime = datetime.datetime.strptime(initial_date+' '+initial_time, '%Y-%m-%d %H:%M')
      final_datetime = datetime.datetime.strptime(final_date+' '+final_time, '%Y-%m-%d %H:%M')

      df = get_data(initial_datetime, final_datetime, data_source, series, interval)
      try:
          # logging.info(df)

          # Download File
          if file_format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
          elif file_format == 'excell':
            output = io.BytesIO()
            df.to_excel(output, index=False)
          elif file_format == 'rinstat':
            output = io.StringIO()
            df.to_csv(output, sep='\t', index=False)

          output.seek(0)

          return Response(
              output,
              mimetype="text/csv",
              headers={"Content-Disposition": "attachment;filename=output.csv"}
          )

      except (psycopg2.DatabaseError, Exception) as error:
        return str(error)

api.add_resource(DataExport, '/data_export')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
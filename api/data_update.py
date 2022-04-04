from __future__ import absolute_import, unicode_literals

import hashlib
import json
import logging
import os
import io
import socket
import subprocess
import pandas as pd
import psycopg2
import pytz
import requests
from tempestas_api import settings


def tables_update():
    #use pandas to import and save WMO csv from github
    #2-02 = WMO Program
    #3-01 = WMO Region
    #3-02 = Country
    #3-04 = Station Type
    #wmoregion, country and stationtype collumns: name, description, notation
    #wmo program collumns: name, description, notation, path
    all_csv = [
        {
        "table": "wx_wmoregion",
        "csv_path":"./csv/3-01.csv"
        },
        {
        "table": "wx_country",
        "csv_path":"./csv/3-02.csv"
        },
        {
        "table": "wx_wmostationtype",
        "csv_path":"https://github.com/wmo-im/wmds/blob/master/tables_en/3-04.csv"
        },
        {
        "table": "wx_wmoprogram",
        "csv_path":"https://github.com/wmo-im/wmds/blob/master/tables_en/2-02.csv"
        }
    ]

    for data in all_csv:

        table = data["table"]
        path = data["csv_path"]
        df = pd.read_csv(path)
        csv_data = df.to_dict('records')

        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:

                if table == "wx_wmoprogram":

                    program_query = ''' 
                        INSERT INTO %(update_table)s(created_at, updated_at, name, notation, description)
                        VALUES (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %(name)s, %(notation)s, %(description)s, %(path)s)
                        ON CONFLICT (name) DO UPDATE
                        SET updated_at = CURRENT_TIMESTAMP,
                        notation = %(notation)s,
                        description = %(description)s
                    '''
                    logging.info(program_query, csv_data)
                    cursor.executemany(program_query, csv_data)
                    
                else:

                    geral_query = ''' 
                        INSERT INTO %(update_table)s(created_at, updated_at, name, notation, description)
                        VALUES (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %(name)s, %(notation)s, %(description)s)
                        ON CONFLICT (name) DO UPDATE
                        SET updated_at = CURRENT_TIMESTAMP,
                        notation = %(notation)s,
                        description = %(description)s
                    '''
                    logging.info(geral_query ,csv_data)
                    cursor.executemany(geral_query, csv_data)

            conn.commit()        
                
tables_update()

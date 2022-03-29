from __future__ import absolute_import, unicode_literals

import hashlib
import json
import logging
import os
import socket
import subprocess
from datetime import datetime, timedelta
from ftplib import FTP, error_perm, error_reply
from time import sleep, time

import cronex
import dateutil.parser
import pandas as pd
import psycopg2
import pytz
import requests
from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from django.db import connection



def import_csv():
    #use pandas to import and save WMO csv from github
    #2-02 = WMO Program
    #3-01 = WMO Region
    #3-02 = Country
    #3-04 = Station Type
    #name, description, notation
    all_csv = [
        {
        "table": "wx_wmoregion",
        "url":"https://github.com/wmo-im/wmds/blob/master/tables_en/3-01.csv"
        },
        {
        "table": "wx_country",
        "url":"https://github.com/wmo-im/wmds/blob/master/tables_en/3-02.csv"
        },
        {
        "table": "wx_wmostationtype",
        "url":"https://github.com/wmo-im/wmds/blob/master/tables_en/3-04.csv"
        },
        {
        "table": "wx_wmoprogram",
        "url":"https://github.com/wmo-im/wmds/blob/master/tables_en/2-02.csv"
        }
    ]

    for data in all_csv:
        data_csv = pd.read_csv(data.url())
        with connection.cursor() as cursor:
            if data.table == "wx_wmoprogram":

                query = ''' '''
                logging.info(query)
                cursor.execute(query)
                pass

            else:

                query2 = ''' '''
                logging.info(query2)
                cursor.execute(query2)
                pass


def update_table(csv_new_data):
    #use psycopg2 to update the table data whit the csv, update set if already exists insert into
    pass

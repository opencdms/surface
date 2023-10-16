#SAT_TX325

import logging
from datetime import datetime, timedelta

import pytz
from django.db import IntegrityError

from wx.decoders.insert_raw_data import insert
from wx.models import VariableFormat, DcpMessages

tz_utc = pytz.timezone("UTC")

ELEMENTS = {
    "BATTERY VOLTAGE": 200,
    "ENC_RH": 4007,
    "RAIN_TOTAL": 0,
    "TEMP_MAX": 16,
    "TEMP_MIN": 14,
    "RH": 30,
    "AVG_WIND_SPEED": 51,
    "AVG_WIND_DIRECTION": 56,
    "WIND_SPEED_MAX": 53,
    "SOLAR_RADIATION": 72,
    "STATION_PRESSURE": 60,
    "SOIL_TEMP": 21,
}


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def parse_line(station_id, header_date, line, interval_lookup_table, records):
    # taking the message and splitting (by blank spaces) into an array
    fields = line.split(",")

    #removing last item fields list
    fields = fields.pop()

    # extracting the hour from the first item in the array
    line_hour = int(fields[0][1:3])

    # extrationg the minute from the first item in the array
    line_minute = int(fields[0][3:5])

    # joining the extrated hour and minute to todays year, day and month to form a complete message
    line_date = datetime(header_date.year, header_date.month, header_date.day, line_hour, line_minute)

    # if hour of measurement is bigger than the transmission hour it is from the previous day
    if line_hour > header_date.hour:
        line_date = line_date - timedelta(days=1)

    line_date = tz_utc.localize(line_date)

    # parsing through the satellite message and transforming all number values into float types
    # note that the utc datatime in the satellite message is also being removed simultaneously 
    values = [parse_float(f) for f in fields[1:]]

    for idx, (k, v) in enumerate(list(zip(list(ELEMENTS.values())[:len(values)], values)), 1):
        if v is not None:
            columns = [
                station_id,                         # station
                k,                                  # element
                interval_lookup_table[str(idx)],    # interval seconds
                line_date,                          # datetime
                v,                                  # value
                None,                               # "quality_flag"
                None,                               # "qc_range_quality_flag"
                None,                               # "qc_range_description"
                None,                               # "qc_step_quality_flag"
                None,                               # "qc_step_description"
                None,                               # "qc_persist_quality_flag"
                None,                               # "qc_persist_description"
                None,                               # "manual_flag"
                None,                               # "consisted"
                False                               # "is_daily"
            ]
            records.append(columns)


def read_data(station_id, dcp_address, config_file, response, err_message):
    print(f'Inside SAT_TX325 decoder - read_data(station_id={station_id}, dcp_address={dcp_address})')

    transmissions = response.split(dcp_address)
    records = []

    dcp_format = 6

    interval_lookup_table = {
        lookup_key: seconds for (lookup_key, seconds) in VariableFormat.objects.filter(
            format_id=dcp_format
        ).values_list('lookup_key', 'interval__seconds')
    }

    for transmission in transmissions[1:]:
        header, *lines = transmission.split(" \r\n")

        print(header)
        print(lines)

        # code can't decode errors like missing transmission spot, so skip error messages
        try:
            header_date = datetime.strptime(header[:11], '%y%j%H%M%S')
            dcp_message = DcpMessages.create(f"{dcp_address}{header}", "\n".join(lines))
            try:
                dcp_message.save()
            except IntegrityError:
                logging.info(f"dcp_message already saved in the database: {header}")

            for line in lines:
                parse_line(station_id, header_date, line, interval_lookup_table, records)

        except Exception as ex:
            _lines = "\n".join(lines)
            logging.error(f"SAT_TX325/CDP Message: Error on decode message for station_id={station_id} "
                          f"dcp_address={dcp_address}\nheader={header}\n"
                          f"lines={_lines}\n{ex}")

    if records:
        print('Inside SAT_TX325 decoder - {0} records downloaded.'.format(len(records)))
        insert(records)
    else:
        print('SAT_TX325 DECODER - NO DATA FOUND - ' + err_message.decode('ascii'))

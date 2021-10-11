import datetime

import pytz

from wx.enums import FlashTypeEnum
from wx.models import Flash


def get_int_from_bytes(bytes, signed=False):
    return int.from_bytes(bytes, byteorder='big', signed=signed)


def read_data(byte_data):
    if byte_data[0] == 56:
        type = FlashTypeEnum.CG.value if byte_data[1] == 0 else FlashTypeEnum.IC.value
        flash_timestamp = get_int_from_bytes(byte_data[2:6])
        flash_nanoseconds = get_int_from_bytes(byte_data[6:10]) / 1e9
        flash_datetime = datetime.datetime.fromtimestamp(flash_timestamp + flash_nanoseconds).astimezone(pytz.UTC)
        latitude = get_int_from_bytes(byte_data[10:14], signed=True) / 1e7
        longitude = get_int_from_bytes(byte_data[14:18], signed=True) / 1e7
        peak_current = get_int_from_bytes(byte_data[18:22], signed=True)
        ic_height = get_int_from_bytes(byte_data[22:24])
        num_sensors = byte_data[24]
        ic_multiplicity = byte_data[25]
        cg_multiplicity = byte_data[26]
        start_timestamp = get_int_from_bytes(byte_data[27:31])
        start_nanoseconds = get_int_from_bytes(byte_data[31:35]) / 1e9
        start_datetime = datetime.datetime.fromtimestamp(start_timestamp + start_nanoseconds).astimezone(pytz.UTC)
        duration = get_int_from_bytes(byte_data[35:39])
        ul_latitude = get_int_from_bytes(byte_data[39:43], signed=True) / 1e7
        ul_longitude = get_int_from_bytes(byte_data[43:47], signed=True) / 1e7
        lr_latitude = get_int_from_bytes(byte_data[47:51], signed=True) / 1e7
        lr_longitude = get_int_from_bytes(byte_data[51:55], signed=True) / 1e7

        print('LIGHTNING DATA - Flash data saved ({0}, {1}).'.format(latitude, longitude))

        Flash.objects.create(type=type,
                             datetime=flash_datetime,
                             latitude=latitude,
                             longitude=longitude,
                             peak_current=peak_current,
                             ic_height=ic_height,
                             num_sensors=num_sensors,
                             ic_multiplicity=ic_multiplicity,
                             cg_multiplicity=cg_multiplicity,
                             start_datetime=start_datetime,
                             duration=duration,
                             ul_latitude=ul_latitude,
                             ul_longitude=ul_longitude,
                             lr_latitude=lr_latitude,
                             lr_longitude=lr_longitude)
    else:
        print('LIGHTNING DATA - Error decoding flash data.')

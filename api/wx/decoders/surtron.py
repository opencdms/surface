# New decoder

import logging
from datetime import datetime, timedelta

import pytz
from django.db import IntegrityError

from wx.decoders.insert_raw_data import insert
from wx.models import VariableFormat, DcpMessages, ElementDecoder

from wx.decoders.surtron_utils import get_config

import os
import re


tz_utc = pytz.timezone("UTC")
tz_bz = pytz.timezone("Etc/GMT+6")

NULL_CHAR = -9999


def get_element_variable_id(element_name):
	# Decoder ID
	SURTRON_ID = 10

	_elementdecoder = ElementDecoder.objects.get(element_name=element_name, decoder=SURTRON_ID)
	variable_id = _elementdecoder.variable_id
	return variable_id

def get_Prefix(msg):
	print('Trying get_Prefix')
	pseudo = msg[1]
	grpID = msg[2]
	delta_time = msg[3]
	return pseudo, grpID, delta_time, msg[4:]

def get_LatLong(msg):
	LatLong = msg[-8:]
	#LATLONG Calculation to be added    
	LATLONG = LatLong    
	return LATLONG, msg[:-8]

def get_Battery(msg):
	Battery = msg[-1]
	BATTERY = (ord(Battery) - 64) * 0.234 + 10.0
	return  BATTERY, msg[:-1]   

def extract_content(msg, Battery=False, LatLong=False, DEBUG = False):
	pseudo, grpID, delta_time, msg = get_Prefix(msg)

	if LatLong:
		LATLONG, msg = get_LatLong(msg)
		if DEBUG:
			print(LATLONG)        
	else:
		LATLONG = 0
	    
	if Battery:
		BATTERY, msg = get_Battery(msg)
		if DEBUG:
			print(f"The appended battery voltage reading is {BATTERY}V .")        

	if pseudo != 'B':
		print("WARNING!!")
		print("Pseudobinary code not in B format, it is in ", pseudo, " instead!")

	return msg, delta_time

############################################################################################

def IsNegative(val):
    if val & 0x20000 == 0x20000:
        val = val - 1
        val = val ^ 0x3FFFF
        val = val * -1
        return val
    else:
        return val

def divide(value,divisor):
    if divisor == 0:
        return value
    else:
        try:
            return value / divisor
        except:
            return value

def settime(base_time,difference):
    record_time = base_time - timedelta(minutes=difference)
    return record_time

def decode_chunk(chunk):
    msb = ord(chunk[0]) - 64 << 12
    if int(ord(chunk[0])) == 63:
        msb = ord(chunk[0]) << 12

    nsb = ord(chunk[1]) - 64 <<  6
    if int(ord(chunk[1])) == 63:
        nsb = ord(chunk[1]) << 6

    lsb = ord(chunk[2]) - 64
    if int(ord(chunk[2])) == 63:
        lsb = ord(chunk[2])

    value = int(msb+nsb+lsb)
    value = IsNegative(value)

    return value

def extract_message_clean(station_id, interval_lookup_table, ID, ID_Decoder, msg, data_time):
	records = []
	chunks = [msg[i:i+3] for i in range(0, len(msg), 3)]

	if len(chunks) != len(ID_Decoder):
		print('The ID_Decoder seems to have differente number of elemets than data points.')

	for counter,chunk in enumerate(chunks):
		if chunk[0]== '/' and chunk[1]== '/' and chunk[2] == '/':
			print(f"{chunk[0]}{chunk[1]}{chunk[2]} --> {settime(data_time,ID_Decoder[counter][2])} {ID_Decoder[counter][0]},{NULL_CHAR}")
		elif len(chunk) == 3:
			value = decode_chunk(chunk)
			if ID_Decoder != None:

				element_name = ID_Decoder[counter][0]
				variable_id = get_element_variable_id(element_name)

				if variable_id != None:
					entry_datetime = settime(data_time,ID_Decoder[counter][2])
					entry_datetime = tz_utc.localize(entry_datetime)
					entry_interval = interval_lookup_table[element_name]
					entry_value = divide(value,ID_Decoder[counter][1])

					columns = [
						station_id,                         # station
						variable_id,						# element
						entry_interval,    					# interval seconds
						entry_datetime,                 	# datetime
						entry_value,						# value
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

					# print(f"{chunk[0]}{chunk[1]}{chunk[2]} --> {SETTIME(data_time,ID_Decoder[counter][2])} {ID_Decoder[counter][0]},{entry_value}")
			else:
				print(f"{chunk[0]}{chunk[1]}{chunk[2]} --> {value}")

	return records

def read_data(station_id, dcp_address, config_data, response, err_message):
	print(f'Inside SURTRON decoder - read_data(station_id={station_id}, dcp_address={dcp_address})')

	transmissions = response.split(dcp_address)
	records = []

	dcp_format = 6

	ID_Decoder, interval_lookup_table = get_config(config_data)

	for transmission in transmissions[1:]:
		header, *lines = transmission.split(" \r\n")
		
		print(header)
		print(lines)

		try:		
			header_date = datetime.strptime(header[:11], '%y%j%H%M%S')
			dcp_message = DcpMessages.create(f"{dcp_address}{header}", "\n".join(lines))
			try:
				dcp_message.save()
			except IntegrityError:
				logging.info(f"dcp_message already saved in the database: {header}")

		    ## Need to test if lines have dada in real scenarios
		    # for line in lines:
		        # parse_line(station_id, header_date, line, interval_lookup_table, records)

			msg = r'bB1F@DX@EI@EI@EI@EI@EI@Cx@Cq@Cp@C~@D^@Dl@DD@Cw@C|@DZ@Dj@E@@AN@@|@@|@@|@@|@@|@Cm@Cr@Cr@Cr@Cr@Cr@@[@@L@@[@@]@@J@@W@AN@BX@CI@A?@Ed@C]@@Q@@R@@Z@@e@@Y@@X@Ai@Bc@Bf@AO@Ax@CR@@m@@i@@q@AP@@q@@s@Ai@Am@B@@A}@AE@CEB\]B\bB\bB\bB\bB\b@@@@@X@HP@G`@A`@@@@A`@A`@A^@@r@@J@@B@BF@BF@BF@BF@BF@BF@MY@LL//////@@@@@@@@@@@@@@@@@@////////////@@WA`fA_AA^_A^PA]eA\~@CL@Ki@Ki@Ki@Ki@Ki////////////'
			msg, delta_time = extract_content(msg, Battery=False, LatLong=False, DEBUG = True)

			time_diff = ord(delta_time) - 64
			data_time = header_date - timedelta(minutes=time_diff)
			data_time = data_time.replace(second=0, microsecond=0)

			curr_records = extract_message_clean(station_id, interval_lookup_table, dcp_address, ID_Decoder, msg, data_time)

			records+=curr_records
		except Exception as ex:
			pass
			_lines = "\n".join(lines)
			logging.error(f"SURTRON/DCP Message: Error on decode message for station_id={station_id} "
							f"dcp_address={dcp_address}\nheader={header}\n"
							f"lines={_lines}\n{ex}")

	if records:
		print('Inside SURTRON decoder - {0} records downloaded.'.format(len(records)))
		insert(records)
	else:
		print('SURTRON DECODER - NO DATA FOUND - ' + err_message.decode('ascii'))
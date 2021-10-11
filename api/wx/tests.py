from datetime import datetime

import pytz
from django.test import TestCase

from wx.decoders.hobo import parse_first_line_header as parse_first_line_header_hobo, \
    parse_second_line_header as parse_second_line_header_hobo, \
    convert_string_2_datetime as convert_string_2_datetime_hobo, get_column_names as get_column_names_hobo, \
    handle_null_field as handle_null_field_hobo
from wx.decoders.toa5 import read_file, parse_first_line_header, parse_second_line_header, convert_string_2_datetime


class IngestTOA5File(TestCase):

    # fixtures = ['fixtures/surface_db.json', ]
    fixtures = ['fixtures/auth.user.json', 'fixtures/wx.json', ]

    # def test_parse_file(self):
    #
    #     filename = '/Users/mbzros/Dropbox/2019/04/04/9922101_yocreek_T_5min2_yHHZM4w.dat'
    #
    #     read_file(filename)

    def test_parse_first_line_header(self):

        first_line = ["TOA5","9922101_yocreek","CR300","6297","CR300.Std.07.04","CPU:9922101_yocreek_modem_06_12_2018.cr300","42220","T_5min2"]

        self.assertEqual(parse_first_line_header(first_line), "9922101")

    def test_parse_second_line_header(self):

        second_line = ["TIMESTAMP","RECORD","BattV_Min","BattV_TMn","Rain_mm_Tot","AirT_C","AirT_C_Max","AirT_C_Min","WS_knot_Avg","WS_knot_Max","WS_knot_TMx","Slr_Wms","WR_km_Tot","Slr_MJ_Tot","WC_C"]

        lookup_table = {0: None, 1: None, 2: None, 3: None, 4: 0, 5: 10, 6: 16, 7: 14, 8: 51, 9: 53, 10: None, 11: 72, 12: 103, 13: 79, 14: 28}

        self.assertEqual(lookup_table, parse_second_line_header(second_line))
    
    def test_read_file(self):

        # filename = '/Users/mbzros/Dropbox/2019/04/04/9901003_belmopan_T_5min1_4pBoxza.dat'
        filename = '/surface/media/documents/2019/04/25/9920101_dangriga_T_5min1_11jzXe3.dat'
        read_file(filename)
    

    def test_convert_string_2_datetime(self):

        text = "2019-04-03 00:10:00"

        date1 = datetime(2019, 4, 3, 0, 10, 0, 0, tzinfo=pytz.UTC)

        self.assertEqual(date1, convert_string_2_datetime(text))

    #######
    # HOBO
    #######

    def test_parse_second_line_header_hobo(self):
        second_line = ["#", "Date Time, GMT-06:00", "AirT_C_MAX_24hr, °C", "AirT_C_MIN_24hr, °C", "Rain_mm_TOT_5min, mm"]

        lookup_table = {0: None, 1: None, 2: 16, 3: 14, 4: 0}

        self.assertEqual(lookup_table, parse_second_line_header_hobo(second_line))

    def test_parse_first_line_header_hobo(self):
        first_line = ["Plot Title: 9910101_Sarteneja_hobo"]

        self.assertEqual(parse_first_line_header_hobo(first_line), "9910101")

    def test_convert_string_2_datetime_hobo(self):
        self.assertEqual(datetime(2018, 12, 13, 0, 0, 0, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/13/18 12:00:00 AM"))

        self.assertEqual(datetime(2018, 12, 14, 10, 0, 0, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/14/18 10:00:00 AM"))

        self.assertEqual(datetime(2018, 12, 13, 23, 59, 59, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/13/18 11:59:59 PM"))

        self.assertEqual(datetime(2018, 12, 13, 0, 59, 59, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/13/18 12:59:59 AM"))

        self.assertEqual(datetime(2018, 12, 13, 1, 59, 59, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/13/18 01:59:59 AM"))

        self.assertEqual(datetime(2018, 12, 13, 1, 59, 59, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/13/18 01:59:59 AM"))

        self.assertEqual(datetime(2018, 12, 13, 12, 0, 0, 0, tzinfo=pytz.UTC),
                         convert_string_2_datetime_hobo("12/13/18 12:00:00 PM"))

    def test_get_column_names_hobo(self):
        linha = ["#", "Date Time, GMT-06:00", "AirT_C_MAX_24hr, °C", "AirT_C_MIN_24hr, °C", "Rain_mm_TOT_5min, mm"]
        column_names = ["#", "Date Time", "AirT_C_MAX_24hr", "AirT_C_MIN_24hr", "Rain_mm_TOT_5min"]

        self.assertEqual(get_column_names_hobo(linha), column_names)

    def test_handle_null_field_hobo(self):
        self.assertEqual(handle_null_field_hobo(''), None)
        self.assertEqual(handle_null_field_hobo('a'), None)
        self.assertEqual(handle_null_field_hobo('3.1'), 3.1)
        self.assertEqual(handle_null_field_hobo('0'), 0.0)

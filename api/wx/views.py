import datetime
import io
import json
import logging
import os
import random
import uuid
from datetime import datetime as datetime_constructor
from datetime import timezone

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import pytz
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.messages.views import SuccessMessageMixin
from django.core.cache import cache
from django.core.exceptions import ObjectDoesNotExist
from django.db import connection
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic.base import TemplateView
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views.generic.list import ListView
from geopandas import geopandas
from material import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.transforms import Bbox
from metpy.interpolate import interpolate_to_grid
from pandas import json_normalize
from rest_framework import viewsets, status, generics, views
from rest_framework.decorators import api_view, permission_classes
from rest_framework.parsers import FileUploadParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from slugify import slugify

from tempestas_api import settings
from wx import serializers, tasks
from wx.decoders import insert_raw_data_pgia
from wx.decoders.hobo import read_file as read_file_hobo
from wx.decoders.toa5 import read_file
from wx.forms import StationForm
from wx.models import AdministrativeRegion, StationFile, Decoder, QualityFlag, DataFile, DataFileStation, \
    DataFileVariable, StationImage, WMOStationType, WMORegion, WMOProgram, StationCommunication
from wx.models import Country, Unit, Station, Variable, DataSource, StationVariable, \
    StationProfile, Document, Watershed, Interval
from wx.utils import get_altitude, get_watershed, get_district, get_interpolation_image, parse_float_value, \
    parse_int_value
from .utils import get_raw_data, get_station_raw_data
from wx.models import VisitType, Technician

logger = logging.getLogger('surface.urls')

# CONSTANT to be used in datetime to milliseconds conversion
EPOCH = datetime_constructor(1970, 1, 1, tzinfo=timezone.utc)


@csrf_exempt
def ScheduleDataExport(request):
    if request.method != 'POST':
        return HttpResponse(status=405)

    json_body = json.loads(request.body)
    station_ids = json_body['stations']  # array with station ids
    data_source = json_body[
        'source']  # one of raw_data, hourly_summary, daily_summary, monthly_summary or yearly_summary
    start_date = json_body['start_datetime']  # in format %Y-%m-%d %H:%M:%S
    end_date = json_body['end_datetime']  # in format %Y-%m-%d %H:%M:%S
    variable_ids = json_body['variables']  # list of obj in format {id: Int, agg: Str}

    data_interval_seconds = None
    if data_source == 'raw_data' and 'data_interval' in json_body:  # a number with the data interval in seconds. Only required for raw_data
        data_interval_seconds = json_body['data_interval']
    elif data_source == 'raw_data':
        data_interval_seconds = 300

    created_data_file_ids = []
    start_date_utc = pytz.UTC.localize(datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S'))
    end_date_utc = pytz.UTC.localize(datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S'))

    if start_date_utc > end_date_utc:
        message = 'The initial date must be greater than final date.'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    days_interval = (end_date_utc - start_date_utc).days

    data_source_dict = {
        "raw_data": "Raw Data",
        "hourly_summary": "Hourly Summary",
        "daily_summary": "Daily Summary",
        "monthly_summary": "Monthly Summary",
        "yearly_summary": "Yearly Summary",
    }

    data_source_description = data_source_dict[data_source]

    prepared_by = None
    if request.user.first_name and request.user.last_name:
        prepared_by = f'{request.user.first_name} {request.user.last_name}'
    else:
        prepared_by = request.user.username

    for station_id in station_ids:
        newfile = DataFile.objects.create(ready=False, initial_date=start_date_utc, final_date=end_date_utc,
                                          source=data_source_description, prepared_by=prepared_by,
                                          interval_in_seconds=data_interval_seconds)
        DataFileStation.objects.create(datafile=newfile, station_id=station_id)

        for variable_id in variable_ids:
            variable = Variable.objects.get(pk=variable_id)
            DataFileVariable.objects.create(datafile=newfile, variable=variable)

        tasks.export_data.delay(station_id, data_source, start_date, end_date, variable_ids, newfile.id)
        created_data_file_ids.append(newfile.id)

    return HttpResponse(created_data_file_ids, status=status.HTTP_200_OK)


@api_view(('GET',))
def DataExportFiles(request):
    files = []
    for df in DataFile.objects.all().order_by('-created_at').values()[:100:1]:
        if df['ready'] and df['ready_at']:
            file_status = {'text': "Ready", 'value': 1}
        elif df['ready_at']:
            file_status = {'text': "Error", 'value': 2}
        else:
            file_status = {'text': "Processing", 'value': 0}

        current_station_name = None
        try:
            current_data_file = DataFileStation.objects.get(datafile_id=df['id'])
            current_station = Station.objects.get(pk=current_data_file.station_id)
            current_station_name = current_station.name
        except ObjectDoesNotExist:
            current_station_name = "Station not found"

        f = {
            'id': df['id'],
            'request_date': df['created_at'],
            'ready_date': df['ready_at'],
            'station': current_station_name,
            'variables': [],
            'status': file_status,
            'initial_date': df['initial_date'],
            'final_date': df['final_date'],
            'source': {'text': df['source'],
                       'value': 0 if df['source'] == 'Raw data' else (1 if df['source'] == 'Hourly summary' else 2)},
            'lines': df['lines'],
            'prepared_by': df['prepared_by']
        }
        if f['ready_date'] is not None:
            f['ready_date'] = f['ready_date']
        for fv in DataFileVariable.objects.filter(datafile_id=df['id']).values():
            f['variables'].append(Variable.objects.filter(pk=fv['variable_id']).values()[0]['name'])
        files.append(f)

    return Response(files, status=status.HTTP_200_OK)


def DownloadDataFile(request):
    file_id = request.GET.get('id', None)
    file_path = os.path.join('/data', 'exported_data', str(file_id) + '.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="text/csv")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    return JsonResponse({}, status=status.HTTP_404_NOT_FOUND)


def DeleteDataFile(request):
    file_id = request.GET.get('id', None)
    df = DataFile.objects.get(pk=file_id)
    DataFileStation.objects.filter(datafile=df).delete()
    DataFileVariable.objects.filter(datafile=df).delete()
    df.delete()
    file_path = os.path.join('/data', 'exported_data', str(file_id) + '.csv')
    if os.path.exists(file_path):
        os.remove(file_path)
    return JsonResponse({}, status=status.HTTP_200_OK)


def GetInterpolationData(request):
    start_datetime = request.GET.get('start_datetime', None)
    end_datetime = request.GET.get('end_datetime', None)
    variable_id = request.GET.get('variable_id', None)
    agg = request.GET.get('agg', "instant")
    source = request.GET.get('source', "raw_data")
    quality_flags = request.GET.get('quality_flags', None)

    where_query = ""
    if source == "raw_data":
        dt_query = "datetime"
        value_query = "measured"
        source_query = "raw_data"
        if quality_flags:
            try:
                [int(qf) for qf in quality_flags.split(',')]
            except ValueError:
                return JsonResponse({"message": "Invalid quality_flags value."}, status=status.HTTP_400_BAD_REQUEST)
            where_query = f" measured != {settings.MISSING_VALUE} AND quality_flag IN ({quality_flags}) AND "
        else:
            where_query = f" measured != {settings.MISSING_VALUE} AND "
    else:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT sampling_operation_id
                FROM wx_variable
                WHERE id=%(variable_id)s;
                """,
                           params={'variable_id': variable_id}
                           )
            sampling_operation = cursor.fetchone()[0]

        if sampling_operation in [6, 7]:
            value_query = "sum_value"
        elif sampling_operation == 3:
            value_query = "min_value"
        elif sampling_operation == 4:
            value_query = "max_value"
        else:
            value_query = "avg_value"

        if source == "hourly":
            dt_query = "datetime"
            source_query = "hourly_summary"

        elif source == "daily":
            dt_query = "day"
            source_query = "daily_summary"

        elif source == "monthly":
            dt_query = "date"
            source_query = "monthly_summary"

        elif source == "yearly":
            dt_query = "date"
            source_query = "yearly_summary"

    if agg == "instant":
        where_query += "variable_id=%(variable_id)s AND " + dt_query + "=%(datetime)s"
        params = {'datetime': start_datetime, 'variable_id': variable_id}
    else:
        where_query += "variable_id=%(variable_id)s AND " + dt_query + " >= %(start_datetime)s AND " + dt_query + " <= %(end_datetime)s"
        params = {'start_datetime': start_datetime, 'end_datetime': end_datetime, 'variable_id': variable_id}

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT a.station_id,b.name,b.code,b.latitude,b.longitude,a.""" + value_query + """ as measured
            FROM """ + source_query + """ a INNER JOIN wx_station b ON a.station_id=b.id
            WHERE """ + where_query + ";",
                       params=params
                       )
        climate_data = {}
        # if agg == "instant":
        raw_data = cursor.fetchall()
        climate_data['data'] = []
        for item in raw_data:
            climate_data['data'].append({
                'station_id': item[0],
                'name': item[1],
                'code': item[2],
                'latitude': item[3],
                'longitude': item[4],
                'measured': item[5],
            })

    if agg != "instant" and len(raw_data) > 0:
        columns = ['station_id', 'name', 'code', 'latitude', 'longitude', 'measured']
        df_climate = json_normalize([
            dict(zip(columns, row))
            for row in raw_data
        ])

        climate_data['data'] = json.loads(
            df_climate.groupby(['station_id', 'name', 'code', 'longitude', 'latitude']).agg(
                agg).reset_index().sort_values('name').to_json(orient="records"))

    return JsonResponse(climate_data)


def GetInterpolationImage(request):
    start_datetime = request.GET.get('start_datetime', None)
    end_datetime = request.GET.get('end_datetime', None)
    variable_id = request.GET.get('variable_id', None)
    cmap = request.GET.get('cmap', 'Spectral_r')
    hres = request.GET.get('hres', 0.01)
    minimum_neighbors = request.GET.get('minimum_neighbors', 1)
    search_radius = request.GET.get('search_radius', 0.7)
    agg = request.GET.get('agg', "instant")
    source = request.GET.get('source', "raw_data")
    vmin = request.GET.get('vmin', 0)
    vmax = request.GET.get('vmax', 30)
    quality_flags = request.GET.get('quality_flags', None)

    stations_df = pd.read_sql_query("""
        SELECT id,name,alias_name,code,latitude,longitude
        FROM wx_station
        WHERE longitude!=0;
        """,
                                    con=connection
                                    )
    stations = geopandas.GeoDataFrame(
        stations_df, geometry=geopandas.points_from_xy(stations_df.longitude, stations_df.latitude))
    stations.crs = 'epsg:4326'

    stands_llat = settings.SPATIAL_ANALYSIS_INITIAL_LATITUDE
    stands_llon = settings.SPATIAL_ANALYSIS_INITIAL_LONGITUDE
    stands_ulat = settings.SPATIAL_ANALYSIS_FINAL_LATITUDE
    stands_ulon = settings.SPATIAL_ANALYSIS_FINAL_LONGITUDE

    where_query = ""
    if source == "raw_data":
        dt_query = "datetime"
        value_query = "measured"
        source_query = "raw_data"
        if quality_flags:
            try:
                [int(qf) for qf in quality_flags.split(',')]
            except ValueError:
                return JsonResponse({"message": "Invalid quality_flags value."}, status=status.HTTP_400_BAD_REQUEST)
            where_query = f" measured != {settings.MISSING_VALUE} AND quality_flag IN ({quality_flags}) AND "
        else:
            where_query = f" measured != {settings.MISSING_VALUE} AND "
    else:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT sampling_operation_id
                FROM wx_variable
                WHERE id=%(variable_id)s;
                """,
                           params={'variable_id': variable_id}
                           )
            sampling_operation = cursor.fetchone()[0]

        if sampling_operation in [6, 7]:
            value_query = "sum_value"
        elif sampling_operation == 3:
            value_query = "min_value"
        elif sampling_operation == 4:
            value_query = "max_value"
        else:
            value_query = "avg_value"

        if source == "hourly":
            dt_query = "datetime"
            source_query = "hourly_summary"

        elif source == "daily":
            dt_query = "day"
            source_query = "daily_summary"

        elif source == "monthly":
            dt_query = "date"
            source_query = "monthly_summary"

        elif source == "yearly":
            dt_query = "date"
            source_query = "yearly_summary"

    if agg == "instant":
        where_query += "variable_id=%(variable_id)s AND " + dt_query + "=%(datetime)s"
        params = {'datetime': start_datetime, 'variable_id': variable_id}
    else:
        where_query += "variable_id=%(variable_id)s AND " + dt_query + " >= %(start_datetime)s AND " + dt_query + " <= %(end_datetime)s"
        params = {'start_datetime': start_datetime, 'end_datetime': end_datetime, 'variable_id': variable_id}

    climate_data = pd.read_sql_query(
        "SELECT station_id,variable_id," + dt_query + "," + value_query + """
        FROM """ + source_query + """
        WHERE """ + where_query + ";",
        params=params,
        con=connection
    )

    if len(climate_data) == 0:
        with open("/surface/static/images/no-interpolated-data.png", "rb") as f:
            img_data = f.read()

        return HttpResponse(img_data, content_type="image/jpeg")

    df_merged = pd.merge(left=climate_data, right=stations, how='left', left_on='station_id', right_on='id')
    df_climate = df_merged[["station_id", dt_query, "longitude", "latitude", value_query]]

    if agg != "instant":
        df_climate = df_climate.groupby(['station_id', 'longitude', 'latitude']).agg(agg).reset_index()

    gx, gy, img = interpolate_to_grid(
        df_climate["longitude"],
        df_climate["latitude"],
        df_climate[value_query],
        interp_type='cressman',
        minimum_neighbors=int(minimum_neighbors),
        hres=float(hres),
        search_radius=float(search_radius),
        boundary_coords={'west': stands_llon, 'east': stands_ulon, 'south': stands_llat, 'north': stands_ulat}
    )

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    fname = str(uuid.uuid4())
    fig.savefig("/surface/static/images/" + fname + ".png", dpi='figure', format='png', transparent=True,
                bbox_inches=Bbox.from_bounds(2, 0, 2.333, 4.013))
    image1 = cv2.imread("/surface/static/images/" + fname + ".png", cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(settings.SPATIAL_ANALYSIS_SHAPE_FILE_PATH, cv2.IMREAD_UNCHANGED)
    image1 = cv2.resize(image1, dsize=(image2.shape[1], image2.shape[0]))
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            image1[i][j][3] = image2[i][j][3]
    cv2.imwrite("/surface/static/images/" + fname + "-output.png", image1)

    with open("/surface/static/images/" + fname + "-output.png", "rb") as f:
        img_data = f.read()

    os.remove("/surface/static/images/" + fname + ".png")
    os.remove("/surface/static/images/" + fname + "-output.png")

    return HttpResponse(img_data, content_type="image/jpeg")


def GetColorMapBar(request):
    start_datetime_req = request.GET.get('start_datetime', '')
    end_datetime_req = request.GET.get('end_datetime', '')
    variable_id = request.GET.get('variable_id', '')
    cmap = request.GET.get('cmap', 'Spectral_r')
    agg = request.GET.get('agg', "instant")
    vmin = request.GET.get('vmin', 0)
    vmax = request.GET.get('vmax', 30)

    try:
        start_datetime = pytz.UTC.localize(datetime.datetime.strptime(start_datetime_req, '%Y-%m-%dT%H:%M:%S.%fZ'))
        end_datetime = pytz.UTC.localize(datetime.datetime.strptime(end_datetime_req, '%Y-%m-%dT%H:%M:%S.%fZ'))

        start_datetime = start_datetime.astimezone(pytz.timezone(settings.TIMEZONE_NAME))
        end_datetime = end_datetime.astimezone(pytz.timezone(settings.TIMEZONE_NAME))
    except ValueError:
        try:
            start_datetime = pytz.UTC.localize(datetime.datetime.strptime(start_datetime_req, '%Y-%m-%d'))
            end_datetime = pytz.UTC.localize(datetime.datetime.strptime(end_datetime_req, '%Y-%m-%d'))

        except ValueError:
            return JsonResponse({"message": "Invalid date format"}, status=status.HTTP_400_BAD_REQUEST)

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT a.name,b.symbol
            FROM wx_variable a INNER JOIN wx_unit b ON a.unit_id=b.id
            WHERE a.id=%(variable_id)s;
            """,
                       params={'variable_id': variable_id}
                       )
        variable = cursor.fetchone()

    fig = plt.figure(figsize=(9, 1.5))
    plt.imshow([[2]], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.2])
    title = variable[0] + ' (' + variable[1] + ') - ' + start_datetime.strftime("%d/%m/%Y %H:%M:%S")
    if agg != 'instant':
        title += ' to ' + end_datetime.strftime("%d/%m/%Y %H:%M:%S")
        title += ' (' + agg + ')'
    cax.set_title(title)
    plt.colorbar(orientation='horizontal', cax=cax)

    FigureCanvas(fig)
    buf = io.BytesIO()
    plt.savefig(buf, dpi='figure', format='png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')

    return response


@csrf_exempt
def InterpolatePostData(request):
    if request.method != 'POST':
        return HttpResponse(status=405)

    stands_llat = settings.SPATIAL_ANALYSIS_INITIAL_LATITUDE
    stands_llon = settings.SPATIAL_ANALYSIS_INITIAL_LONGITUDE
    stands_ulat = settings.SPATIAL_ANALYSIS_FINAL_LATITUDE
    stands_ulon = settings.SPATIAL_ANALYSIS_FINAL_LONGITUDE

    json_body = json.loads(request.body)
    parameters = json_body['parameters']
    vmin = json_body['vmin']
    vmax = json_body['vmax']
    df_climate = json_normalize(json_body['data'])
    try:
        df_climate = df_climate[["station_id", "longitude", "latitude", "measured"]]
    except KeyError:
        return HttpResponse("no-interpolated-data.png")

    gx, gy, img = interpolate_to_grid(
        df_climate["longitude"],
        df_climate["latitude"],
        df_climate["measured"],
        interp_type='cressman',
        minimum_neighbors=int(parameters["minimum_neighbors"]),
        hres=float(parameters["hres"]),
        search_radius=float(parameters["search_radius"]),
        boundary_coords={'west': stands_llon, 'east': stands_ulon, 'south': stands_llat, 'north': stands_ulat}
    )

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, origin='lower', cmap=parameters["cmap"]["value"], vmin=vmin, vmax=vmax)
    for filename in os.listdir('/surface/static/images'):
        try:
            if datetime.datetime.now() - datetime.datetime.strptime(filename.split('_')[0],
                                                                    "%Y-%m-%dT%H:%M:%SZ") > datetime.timedelta(
                minutes=5):
                os.remove('/surface/static/images/' + filename)
        except ValueError:
            continue
    fname = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ_") + str(uuid.uuid4())
    fig.savefig("/surface/static/images/" + fname + ".png", dpi='figure', format='png', transparent=True,
                bbox_inches=Bbox.from_bounds(2, 0, 2.333, 4.013))
    image1 = cv2.imread("/surface/static/images/" + fname + ".png", cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread('/surface/static/images/blz_shape.png', cv2.IMREAD_UNCHANGED)
    image1 = cv2.resize(image1, dsize=(image2.shape[1], image2.shape[0]))
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            image1[i][j][3] = image2[i][j][3]
    cv2.imwrite("/surface/static/images/" + fname + "-output.png", image1)

    return HttpResponse(fname + "-output.png")


@permission_classes([IsAuthenticated])
def GetImage(request):
    image = request.GET.get('image', None)
    try:
        with open("/surface/static/images/" + image, "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpeg")
    except IOError:
        red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
        response = HttpResponse(content_type="image/jpeg")
        red.save(response, "JPEG")
        return response


from django.db.models.functions import Cast
from django.db.models import IntegerField


@permission_classes([IsAuthenticated])
def MonthlyFormView(request):
    template = loader.get_template('wx/monthly_form.html')

    station_list = Station.objects.filter(is_automatic=False, is_active=True)
    station_list = station_list.values('id', 'name', 'code')
    
    # for station in station_list:
    #     station['code'] = int(station['code'])

    context = {'station_list': station_list}

    return HttpResponse(template.render(context, request))


class PGIAReportView(LoginRequiredMixin, TemplateView):
    template_name = "wx/pgiareport.html"


@permission_classes([IsAuthenticated])
def DataCaptureView(request):
    template = loader.get_template('wx/data_capture.html')
    return HttpResponse(template.render({}, request))


class DataExportView(LoginRequiredMixin, TemplateView):
    template_name = "wx/data_export.html"

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.select_related('profile').all()
        context['variable_list'] = Variable.objects.select_related('unit').all()

        context['station_profile_list'] = StationProfile.objects.all()
        context['station_watershed_list'] = Watershed.objects.all()
        context['station_district_list'] = AdministrativeRegion.objects.all()

        interval_list = Interval.objects.filter(seconds__lte=3600).order_by('seconds')
        context['interval_list'] = interval_list

        return self.render_to_response(context)


class CountryViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = Country.objects.all().order_by("name")
    serializer_class = serializers.CountrySerializer


class UnitViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = Unit.objects.all().order_by("name")
    serializer_class = serializers.UnitSerializer


class DataSourceViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = DataSource.objects.all().order_by("name")
    serializer_class = serializers.DataSourceSerializer


class VariableViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = Variable.objects.all().order_by("name")
    serializer_class = serializers.VariableSerializer


class StationMetadataViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = Station.objects.all()
    serializer_class = serializers.StationMetadataSerializer

    def get_serializer_class(self):
        if self.request.method in ['GET']:
            return serializers.StationSerializerRead
        return serializers.StationMetadataSerializer


class StationViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = Station.objects.all()

    # queryset = Station.objects.all().select_related("country").order_by("name")
    # def put(self, request, *args, **kwargs):
    #     station_object = Station.objects.get()
    #     data = request.data

    #     station_object.save()
    # serializer = serializers.StationSerializerWrite
        

    def get_serializer_class(self):
        if self.request.method in ['GET']:
            return serializers.StationSerializerRead

        return serializers.StationSerializerWrite


class StationSimpleViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = Station.objects.all()
    serializer_class = serializers.StationSimpleSerializer


class StationVariableViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = StationVariable.objects.all().order_by("variable")
    serializer_class = serializers.StationVariableSerializer

    def get_queryset(self):
        queryset = StationVariable.objects.all()

        station_id = self.request.query_params.get('station_id', None)

        if station_id is not None:
            queryset = queryset.filter(station__id=station_id)

        return queryset


class StationProfileViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = StationProfile.objects.all().order_by("name")
    serializer_class = serializers.StationProfileSerializer



class DocumentViewSet(views.APIView):
    permission_classes = (IsAuthenticated,)
    parser_class = (FileUploadParser,)
    queryset = Document.objects.all()
    serializer_class = serializers.DocumentSerializer
    available_decoders = {'HOBO': read_file_hobo, 'TOA5': read_file}
    decoders = Decoder.objects.all().exclude(name='DCP TEXT').exclude(name='NESA')

    def put(self, request, format=None):
        selected_decoder = 'TOA5'

        if 'decoder' in request.data.keys():
            selected_decoder = request.data['decoder']

        serializer = serializers.DocumentSerializer(data=request.data)

        if serializer.is_valid():
            document = serializer.save()
            self.available_decoders[selected_decoder].delay(document.file.path)
            return Response({"message": "FIle uploaded successfully!"}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, format=None):
        return self.put(request, format=None)


class AdministrativeRegionViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = AdministrativeRegion.objects.all().order_by("name")
    serializer_class = serializers.AdministrativeRegionSerializer


@api_view(['GET'])
def station_telemetry_data(request, date):
    mock = {
        'temperature': {
            'min': 10,
            'max': 13,
            'avg': 16
        },
        'relativeHumidity': {
            'min': 10,
            'max': 13,
            'avg': 16
        },
        'precipitation': {
            'current': 123
        },
        'windDirection': {
            'current': 'SW'
        },
        'windSpeed': {
            'current': 11,
            'max': 12
        },
        'windGust': {
            'current': 12,
            'max': 11
        },
        'solarRadiation': {
            'current': 12
        },
        'atmosphericPressure': {
            'current': 11
        }
    }

    data = {
        'latest': mock,
        'last24': mock,
        'current': mock,
    }

    return Response(data)


def raw_data_list(request):
    search_type = request.GET.get('search_type', None)
    search_value = request.GET.get('search_value', None)
    search_value2 = request.GET.get('search_value2', None)
    search_date_start = request.GET.get(
        'search_date_start',
        default=(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
    )
    search_date_end = request.GET.get(
        'search_date_end',
        default=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    )
    sql_string = ""

    response = {
        'results': [],
        'messages': [],
    }

    try:
        start_date = datetime.datetime.strptime(search_date_start, '%Y-%m-%dT%H:%M:%SZ')
        end_date = datetime.datetime.strptime(search_date_end, '%Y-%m-%dT%H:%M:%SZ')

    except ValueError:
        message = 'Invalid date format. Expected YYYY-MM-DDTHH:MI:SSZ'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    delta = end_date - start_date

    if delta.days > 8:  # Restrict queries to max seven days
        message = 'Interval between start date and end date is greater than one week.'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    if search_type in ['station', 'stationvariable']:
        try:
            station = Station.objects.get(code=search_value)
        except ObjectDoesNotExist:
            station = Station.objects.get(pk=search_value)
        finally:
            search_value = station.id

    if search_type is not None and search_type == 'variable':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   c.name,
                   c.symbol,
                   a.measured,
                   a.datetime,
                   q.symbol as quality_flag,
                   b.variable_type,
                   a.code
            FROM raw_data a
            JOIN wx_variable b ON a.variable_id=b.id
            LEFT JOIN wx_unit c ON b.unit_id=c.id
            JOIN wx_qualityflag q ON a.quality_flag=q.id
        WHERE b.id = %s 
          AND datetime >= %s 
          AND datetime <= %s
        """

    if search_type is not None and search_type == 'station':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   c.name,
                   c.symbol,
                   a.measured,
                   a.datetime,
                   q.symbol as quality_flag,
                   b.variable_type,
                   a.code
            FROM raw_data a
            JOIN wx_variable b ON a.variable_id=b.id
            LEFT JOIN wx_unit c ON b.unit_id=c.id
            JOIN wx_qualityflag q ON a.quality_flag=q.id
            WHERE station_id=%s AND datetime >= %s AND datetime <= %s"""

    if search_type is not None and search_type == 'stationvariable':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   c.name,
                   c.symbol,
                   a.measured,
                   a.datetime,
                   q.symbol as quality_flag,
                   b.variable_type,
                   a.code
            FROM raw_data a
            JOIN wx_variable b ON a.variable_id=b.id
            LEFT JOIN wx_unit c ON b.unit_id=c.id
            JOIN wx_qualityflag q ON a.quality_flag=q.id
            WHERE station_id=%s AND variable_id=%s AND datetime >= %s AND datetime <= %s"""

    if sql_string:
        sql_string += " ORDER BY datetime"

        with connection.cursor() as cursor:

            if search_type is not None and search_type == 'stationvariable':

                cursor.execute(sql_string, [search_value, search_value2, search_date_start, search_date_end])

            else:

                cursor.execute(sql_string, [search_value, search_date_start, search_date_end])

            rows = cursor.fetchall()

            for row in rows:

                if row[9] is not None and row[9].lower() == 'code':
                    value = row[10]
                else:
                    value = round(row[6], 2)

                obj = {
                    'station': row[0],
                    'date': row[7],
                    'value': value,
                    'variable': {
                        'symbol': row[2],
                        'name': row[3],
                        'unit_name': row[4],
                        'unit_symbol': row[5]
                    }
                }

                response['results'].append(obj)

            if response['results']:
                return JsonResponse(response, status=status.HTTP_200_OK)

    return JsonResponse(data={"message": "No data found."}, status=status.HTTP_404_NOT_FOUND)


def hourly_summary_list(request):
    search_type = request.GET.get('search_type', None)
    search_value = request.GET.get('search_value', None)
    search_value2 = request.GET.get('search_value2', None)
    search_date_start = request.GET.get(
        'search_date_start',
        default=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    )
    search_date_end = request.GET.get(
        'search_date_end',
        default=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    )
    sql_string = ""

    response = {
        'results': [],
        'messages': [],
    }

    try:
        start_date = datetime.datetime.strptime(search_date_start, '%Y-%m-%dT%H:%M:%SZ')
        end_date = datetime.datetime.strptime(search_date_end, '%Y-%m-%dT%H:%M:%SZ')

    except ValueError:
        message = 'Invalid date format. Expected YYYY-MM-DDTHH:MI:SSZ'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    delta = end_date - start_date

    if delta.days > 32:  # Restrict queries to max 31 days
        message = 'Interval between start date and end date is greater than one month.'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    if search_type in ['station', 'stationvariable']:
        try:
            station = Station.objects.get(code=search_value)
        except ObjectDoesNotExist:
            station = Station.objects.get(pk=search_value)
        finally:
            search_value = station.id

    if search_type is not None and search_type == 'variable':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   b.sampling_operation_id,
                   c.name,
                   c.symbol,
                   min_value,
                   max_value,
                   avg_value,
                   sum_value,
                   num_records,
                   datetime as data
              FROM hourly_summary a
        INNER JOIN wx_variable b ON a.variable_id=b.id
        INNER JOIN wx_unit c ON b.unit_id=c.id
        WHERE b.id=%s AND datetime >= %s AND datetime <= %s"""

    if search_type is not None and search_type == 'station':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   b.sampling_operation_id,
                   c.name,
                   c.symbol,
                   min_value,
                   max_value,
                   avg_value,
                   sum_value,
                   num_records,
                   datetime as data
              FROM hourly_summary a
        INNER JOIN wx_variable b ON a.variable_id=b.id
        INNER JOIN wx_unit c ON b.unit_id=c.id
             WHERE station_id=%s AND datetime >= %s AND datetime <= %s"""

    if search_type is not None and search_type == 'stationvariable':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   b.sampling_operation_id,
                   c.name,
                   c.symbol,
                   min_value,
                   max_value,
                   avg_value,
                   sum_value,
                   num_records,
                   datetime as data
              FROM hourly_summary a
        INNER JOIN wx_variable b ON a.variable_id=b.id
        INNER JOIN wx_unit c ON b.unit_id=c.id
             WHERE station_id=%s AND variable_id=%s AND datetime >= %s AND datetime <= %s"""

    if sql_string:
        sql_string += " ORDER BY datetime"

        with connection.cursor() as cursor:

            if search_type is not None and search_type == 'stationvariable':

                cursor.execute(sql_string, [search_value, search_value2, search_date_start, search_date_end])

            else:

                cursor.execute(sql_string, [search_value, search_date_start, search_date_end])

            rows = cursor.fetchall()

            for row in rows:

                value = None

                if row[4] in [1, 2]:
                    value = row[9]

                elif row[4] == 3:
                    value = row[7]

                elif row[4] == 4:
                    value = row[8]

                elif row[4] == 6:
                    value = row[10]

                else:
                    value = row[10]

                if value is None:

                    print('variable {} does not have supported sampling operation {}'.format(row[1], row[4]))

                else:

                    obj = {
                        'station': row[0],
                        'date': row[12],
                        'value': round(value, 2),
                        'min': round(row[7], 2),
                        'max': round(row[8], 2),
                        'avg': round(row[9], 2),
                        'sum': round(row[10], 2),
                        'count': round(row[11], 2),
                        'variable': {
                            'symbol': row[2],
                            'name': row[3],
                            'unit_name': row[5],
                            'unit_symbol': row[6],
                        }
                    }

                    response['results'].append(obj)

            if response['results']:
                return JsonResponse(response, status=status.HTTP_200_OK)

    return JsonResponse(data=response)


def daily_summary_list(request):
    search_type = request.GET.get('search_type', None)
    search_value = request.GET.get('search_value', None)
    search_value2 = request.GET.get('search_value2', None)
    search_date_start = request.GET.get(
        'search_date_start',
        default=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    )
    search_date_end = request.GET.get(
        'search_date_end',
        default=datetime.datetime.now().strftime('%Y-%m-%d')
    )
    sql_string = ""

    response = {
        'results': [],
        'messages': [],
    }

    try:
        start_date = datetime.datetime.strptime(search_date_start, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(search_date_end, '%Y-%m-%d')

    except ValueError:
        message = 'Invalid date format. Expected YYYY-MM-DD'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    delta = end_date - start_date

    if delta.days > 400:  # Restrict queries to max 400 days
        message = 'Interval between start date and end date is greater than 13 months.'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    if search_type in ['station', 'stationvariable']:
        try:
            station = Station.objects.get(code=search_value)
        except ObjectDoesNotExist:
            station = Station.objects.get(pk=search_value)
        finally:
            search_value = station.id

    if search_type is not None and search_type == 'variable':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   b.sampling_operation_id,
                   c.name,
                   c.symbol,
                   min_value,
                   max_value,
                   avg_value,
                   sum_value,
                   day,
                   num_records
              FROM daily_summary a
        INNER JOIN wx_variable b ON a.variable_id=b.id
        INNER JOIN wx_unit c ON b.unit_id=c.id
        WHERE b.id = %s 
          AND day >= %s 
          AND day <= %s"""

    if search_type is not None and search_type == 'station':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   b.sampling_operation_id,
                   c.name,
                   c.symbol,
                   min_value,
                   max_value,
                   avg_value,
                   sum_value,
                   day,
                   num_records
              FROM daily_summary a
        INNER JOIN wx_variable b ON a.variable_id=b.id
        INNER JOIN wx_unit c ON b.unit_id=c.id
             WHERE station_id=%s AND day >= %s AND day <= %s"""

    if search_type is not None and search_type == 'stationvariable':
        sql_string = """
            SELECT station_id,
                   variable_id,
                   b.symbol,
                   b.name,
                   b.sampling_operation_id,
                   c.name,
                   c.symbol,
                   min_value,
                   max_value,
                   avg_value,
                   sum_value,
                   day,
                   num_records
              FROM daily_summary a
        INNER JOIN wx_variable b ON a.variable_id=b.id
        INNER JOIN wx_unit c ON b.unit_id=c.id
             WHERE station_id=%s AND variable_id=%s AND day >= %s AND day <= %s"""

    if sql_string:
        sql_string += " ORDER BY day"

        with connection.cursor() as cursor:

            if search_type is not None and search_type == 'stationvariable':
                cursor.execute(sql_string, [search_value, search_value2, search_date_start, search_date_end])
            else:
                cursor.execute(sql_string, [search_value, search_date_start, search_date_end])

            rows = cursor.fetchall()

            for row in rows:

                value = None

                if row[4] in [1, 2]:
                    value = row[9]

                elif row[4] == 3:
                    value = row[7]

                elif row[4] == 4:
                    value = row[8]

                elif row[4] == 6:
                    value = row[10]

                else:
                    value = row[10]

                if value is not None:

                    obj = {
                        'station': row[0],
                        'date': row[11],
                        'value': round(value, 2),
                        'min': round(row[7], 2),
                        'max': round(row[8], 2),
                        'avg': round(row[9], 2),
                        'total': round(row[10], 2),
                        'count': row[12],
                        'variable': {
                            'symbol': row[2],
                            'name': row[3],
                            'unit_name': row[5],
                            'unit_symbol': row[6],
                        }
                    }

                    response['results'].append(obj)

                else:
                    JsonResponse(data={
                        "message": 'variable {} does not have supported sampling operation {}'.format(row[1], row[4])},
                        status=status.HTTP_400_BAD_REQUEST)

            if response['results']:
                return JsonResponse(response, status=status.HTTP_200_OK)

    return JsonResponse(data=response)


def monthly_summary_list(request):
    search_type = request.GET.get('search_type', None)
    search_value = request.GET.get('search_value', None)
    search_value2 = request.GET.get('search_value2', None)
    search_date_start = request.GET.get(
        'search_date_start',
        default=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    )
    search_date_end = request.GET.get(
        'search_date_end',
        default=datetime.datetime.now().strftime('%Y-%m-%d')
    )

    sql_string = ""

    response = {
        'count': -999,
        'next': None,
        'previous': None,
        'results': []
    }

    try:
        start_date = datetime.datetime.strptime(search_date_start, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(search_date_end, '%Y-%m-%d')

    except ValueError:
        message = 'Invalid date format. Expected YYYY-MM-DD'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    if search_type in ['station', 'stationvariable']:
        try:
            station = Station.objects.get(code=search_value)
        except ObjectDoesNotExist:
            station = Station.objects.get(pk=search_value)
        finally:
            search_value = station.id

    if search_type is not None and search_type == 'variable':
        sql_string = """
            SELECT  station_id,
                    variable_id,
                    b.symbol,
                    b.name,
                    b.sampling_operation_id,
                    c.name,
                    c.symbol,
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    date::date,
                    num_records
            FROM monthly_summary a
            JOIN wx_variable b ON a.variable_id=b.id 
            JOIN wx_unit c ON b.unit_id=c.id 
            WHERE b.id = %s
              AND date >= %s 
              AND date <= %s
        """

    if search_type is not None and search_type == 'station':
        sql_string = """
            SELECT  station_id,
                    variable_id,
                    b.symbol, 
                    b.name, 
                    b.sampling_operation_id,
                    c.name, 
                    c.symbol, 
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    date::date,
                    num_records
            FROM monthly_summary a 
            JOIN wx_variable b ON a.variable_id=b.id
            JOIN wx_unit c ON b.unit_id=c.id 
            WHERE station_id = %s 
              AND date >= %s AND date <= %s
        """

    if search_type is not None and search_type == 'stationvariable':
        sql_string = """
            SELECT  station_id,
                    variable_id,
                    b.symbol, 
                    b.name, 
                    b.sampling_operation_id,
                    c.name, 
                    c.symbol, 
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    date::date,
                    num_records
            FROM monthly_summary a 
            JOIN wx_variable b ON a.variable_id=b.id
            JOIN wx_unit c ON b.unit_id=c.id 
            WHERE station_id = %s 
              AND variable_id = %s
              AND date >= %s AND date <= %s
        """

    if sql_string:
        sql_string += " ORDER BY month"

        with connection.cursor() as cursor:

            if search_type is not None and search_type == 'stationvariable':
                cursor.execute(sql_string, [search_value, search_value2, start_date, end_date])
            else:
                cursor.execute(sql_string, [search_value, start_date, end_date])

            rows = cursor.fetchall()

            for row in rows:

                value = None

                if row[4] in [1, 2]:
                    value = row[9]

                elif row[4] == 3:
                    value = row[7]

                elif row[4] == 4:
                    value = row[8]

                elif row[4] == 6:
                    value = row[10]

                else:
                    value = row[10]

                if value is not None:

                    obj = {
                        'station': row[0],
                        'date': row[11],
                        'value': round(value, 2),
                        'min': round(row[7], 2),
                        'max': round(row[8], 2),
                        'avg': round(row[9], 2),
                        'total': round(row[10], 2),
                        'count': row[12],
                        'variable': {
                            'symbol': row[2],
                            'name': row[3],
                            'unit_name': row[5],
                            'unit_symbol': row[6],
                        }
                    }

                    response['results'].append(obj)

                else:
                    JsonResponse(data={
                        "message": 'variable {} does not have supported sampling operation {}'.format(row[1], row[4])},
                        status=status.HTTP_400_BAD_REQUEST)

            if response['results']:
                return JsonResponse(response, status=status.HTTP_200_OK)

    return JsonResponse(data=response)


def yearly_summary_list(request):
    search_type = request.GET.get('search_type', None)
    search_value = request.GET.get('search_value', None)
    search_value2 = request.GET.get('search_value2', None)
    search_date_start = request.GET.get(
        'search_date_start',
        default=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    )
    search_date_end = request.GET.get(
        'search_date_end',
        default=datetime.datetime.now().strftime('%Y-%m-%d')
    )

    sql_string = ""

    response = {
        'count': -999,
        'next': None,
        'previous': None,
        'results': []
    }

    try:
        start_date = datetime.datetime.strptime(search_date_start, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(search_date_end, '%Y-%m-%d')

    except ValueError:
        message = 'Invalid date format. Expected YYYY-MM-DD'
        return JsonResponse(data={"message": message}, status=status.HTTP_400_BAD_REQUEST)

    if search_type in ['station', 'stationvariable']:
        try:
            station = Station.objects.get(code=search_value)
        except ObjectDoesNotExist:
            station = Station.objects.get(pk=search_value)
        finally:
            search_value = station.id

    if search_type is not None and search_type == 'variable':
        sql_string = """
            SELECT  station_id,
                    variable_id,
                    b.symbol,
                    b.name,
                    b.sampling_operation_id,
                    c.name,
                    c.symbol,
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    date::date,
                    num_records
            FROM yearly_summary a
            JOIN wx_variable b ON a.variable_id=b.id 
            JOIN wx_unit c ON b.unit_id=c.id 
            WHERE b.id = %s
              AND date >= %s 
              AND date <= %s
        """

    if search_type is not None and search_type == 'station':
        sql_string = """
            SELECT  station_id,
                    variable_id,
                    b.symbol, 
                    b.name, 
                    b.sampling_operation_id,
                    c.name, 
                    c.symbol, 
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    date::date,
                    num_records
            FROM yearly_summary a 
            JOIN wx_variable b ON a.variable_id=b.id
            JOIN wx_unit c ON b.unit_id=c.id 
            WHERE station_id = %s 
              AND date >= %s AND date <= %s
        """

    if search_type is not None and search_type == 'stationvariable':
        sql_string = """
            SELECT  station_id,
                    variable_id,
                    b.symbol, 
                    b.name, 
                    b.sampling_operation_id,
                    c.name, 
                    c.symbol, 
                    min_value,
                    max_value,
                    avg_value,
                    sum_value,
                    date::date,
                    num_records
            FROM yearly_summary a 
            JOIN wx_variable b ON a.variable_id=b.id
            JOIN wx_unit c ON b.unit_id=c.id 
            WHERE station_id = %s 
              AND variable_id = %s
              AND date >= %s AND date <= %s
        """

    if sql_string:
        sql_string += " ORDER BY year"

        with connection.cursor() as cursor:

            if search_type is not None and search_type == 'stationvariable':
                cursor.execute(sql_string, [search_value, search_value2, start_date, end_date])
            else:
                cursor.execute(sql_string, [search_value, start_date, end_date])

            rows = cursor.fetchall()

            for row in rows:

                value = None

                if row[4] in [1, 2]:
                    value = row[9]

                elif row[4] == 3:
                    value = row[7]

                elif row[4] == 4:
                    value = row[8]

                elif row[4] == 6:
                    value = row[10]

                else:
                    value = row[10]

                if value is not None:

                    obj = {
                        'station': row[0],
                        'date': row[11],
                        'value': round(value, 2),
                        'min': round(row[7], 2),
                        'max': round(row[8], 2),
                        'avg': round(row[9], 2),
                        'total': round(row[10], 2),
                        'count': row[12],
                        'variable': {
                            'symbol': row[2],
                            'name': row[3],
                            'unit_name': row[5],
                            'unit_symbol': row[6],
                        }
                    }

                    response['results'].append(obj)

                else:
                    JsonResponse(data={
                        "message": 'variable {} does not have supported sampling operation {}'.format(row[1], row[4])},
                        status=status.HTTP_400_BAD_REQUEST)

            if response['results']:
                return JsonResponse(response, status=status.HTTP_200_OK)

    return JsonResponse(data=response)


@api_view(['GET'])
def station_geo_features(request, lon, lat):
    longitude = float(lon)
    latitude = float(lat)

    altitude = get_altitude(longitude, latitude)

    watershed = get_watershed(longitude, latitude)

    district = get_district(longitude, latitude)

    data = {
        'elevation': altitude,
        'watershed': watershed,
        'country': 'Belize',
        'administrative_region': district,
        'longitude': longitude,
        'latitude': latitude,
    }

    return Response(data)


def get_last24_data(station_id):
    result = []
    max_date = None

    query = """
        SELECT
            last24h.datetime,
            var.name,
            var.symbol,
            var.sampling_operation_id,
            unit.name,
            unit.symbol,
            last24h.min_value,
            last24h.max_value,
            last24h.avg_value,
            last24h.sum_value,
            last24h.latest_value
        FROM
            last24h_summary last24h
        INNER JOIN
            wx_variable var ON last24h.variable_id=var.id
        INNER JOIN
            wx_unit unit ON var.unit_id=unit.id
        WHERE
            last24h.station_id=%s
        ORDER BY var.name"""

    with connection.cursor() as cursor:

        cursor.execute(query, [station_id])

        rows = cursor.fetchall()

        for row in rows:

            value = None

            if row[3] == 1:
                value = row[10]

            elif row[3] == 2:
                value = row[8]

            elif row[3] == 3:
                value = row[6]

            elif row[3] == 4:
                value = row[7]

            elif row[3] == 6:
                value = row[9]

            if value is None:
                print('variable {} does not have supported sampling operation {}'.format(row[1], row[3]))

            obj = {
                'value': value,
                'variable': {
                    'name': row[1],
                    'symbol': row[2],
                    'unit_name': row[4],
                    'unit_symbol': row[5]
                }
            }
            result.append(obj)

        max_date = cache.get('last24h_summary_last_run', None)

    return result, max_date


def get_latest_data(station_id):
    result = []
    max_date = None

    query = """
        SELECT CASE WHEN var.variable_type ilike 'code' THEN latest.last_data_code ELSE latest.last_data_value::varchar END as value,
               latest.last_data_datetime,
               var.name,
               var.symbol,
               unit.name,
               unit.symbol
        FROM wx_stationvariable latest
        INNER JOIN wx_variable var ON latest.variable_id=var.id
        LEFT JOIN wx_unit unit ON var.unit_id=unit.id
        WHERE latest.station_id=%s 
          AND latest.last_data_value is not null
          AND latest.last_data_datetime = ( SELECT MAX(most_recent.last_data_datetime)
                                            FROM wx_stationvariable most_recent
                                            WHERE most_recent.station_id=latest.station_id 
                                                AND most_recent.last_data_value is not null)
        ORDER BY var.name
        """

    with connection.cursor() as cursor:

        cursor.execute(query, [station_id])

        rows = cursor.fetchall()

        for row in rows:
            obj = {
                'value': row[0],
                'variable': {
                    'name': row[2],
                    'symbol': row[3],
                    'unit_name': row[4],
                    'unit_symbol': row[5]
                }
            }
            result.append(obj)

        if rows:
            max_date = rows[-1][1]

    return result, max_date


def get_current_data(station_id):
    result = []
    max_date = None
    parameter_timezone = pytz.timezone(settings.TIMEZONE_NAME)
    today = datetime.datetime.now().astimezone(parameter_timezone).date()

    query = """
        SELECT current.day,
               var.name,
               var.symbol,
               var.sampling_operation_id,
               unit.name,
               unit.symbol,
               current.min_value,
               current.max_value,
               current.avg_value,
               current.sum_value
        FROM daily_summary current
        INNER JOIN wx_variable var ON current.variable_id=var.id
        INNER JOIN wx_unit unit ON var.unit_id=unit.id
        WHERE current.station_id=%s and current.day=%s
        ORDER BY current.day, var.name
    """

    with connection.cursor() as cursor:

        cursor.execute(query, [station_id, today])

        rows = cursor.fetchall()

        for row in rows:

            value = None

            if row[3] in (1, 2):
                value = row[8]

            elif row[3] == 3:
                value = row[6]

            elif row[3] == 4:
                value = row[7]

            elif row[3] == 6:
                value = row[9]

            if value is None:
                print('variable {} does not have supported sampling operation {}'.format(row[1], row[3]))

            obj = {
                'value': value,
                'variable': {
                    'name': row[1],
                    'symbol': row[2],
                    'unit_name': row[4],
                    'unit_symbol': row[5]
                }
            }
            result.append(obj)

        max_date = cache.get('daily_summary_last_run', None)

    return result, max_date


@api_view(['GET'])
def livedata(request, code):
    try:
        station = Station.objects.get(code=code)
    except ObjectDoesNotExist as e:
        station = Station.objects.get(pk=code)
    finally:
        id = station.id

    past24h_data, past24h_max_date = get_last24_data(station_id=id)
    latest_data, latest_max_date = get_latest_data(station_id=id)
    current_data, current_max_date = get_current_data(station_id=id)

    station_data = serializers.StationSerializerRead(station).data

    return Response(
        {
            'station': station_data,
            'station_name': station.name,
            'station_id': station.id,
            'past24h': past24h_data,
            'past24h_last_update': past24h_max_date,
            'latest': latest_data,
            'latest_last_update': latest_max_date,
            'currentday': current_data,
            'currentday_last_update': current_max_date,
        }
        , status=status.HTTP_200_OK)


class WatershedList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = serializers.WatershedSerializer
    queryset = Watershed.objects.all().order_by("watershed")


class StationCommunicationList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = serializers.StationCommunicationSerializer
    queryset = StationCommunication.objects.all()


class DecoderList(generics.ListAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = serializers.DecoderSerializer
    queryset = Decoder.objects.all().order_by("name")


class QualityFlagList(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticated,)
    serializer_class = serializers.QualityFlagSerializer
    queryset = QualityFlag.objects.all().order_by("name")


def qc_list(request):
    if request.method == 'GET':
        station_id = request.GET.get('station_id', None)
        variable_id = request.GET.get('variable_id', None)
        start_date = request.GET.get('start_date', None)
        end_date = request.GET.get('end_date', None)

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id is None:
            JsonResponse(data={"message": "'variable_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        sql_string = ""
        where_parameters = [station_id, variable_id]

        response = {
            'count': -999,
            'next': None,
            'previous': None,
            'results': []
        }

        sql_string = """SELECT value.datetime
                            ,value.measured
                            ,value.consisted
                            ,value.quality_flag
                            ,value.manual_flag
                            ,value.station_id
                            ,value.variable_id
                            ,value.remarks
                            ,value.ml_flag
                        FROM raw_data as value
                        WHERE value.station_id=%s
                        AND value.variable_id=%s
                    """

        if start_date is not None and end_date is not None:
            where_parameters.append(start_date)
            where_parameters.append(end_date)
            sql_string += " AND value.datetime >= %s AND value.datetime <= %s"

        elif start_date is not None:
            where_parameters.append(start_date)
            sql_string += " AND value.datetime >= %s"

        elif end_date is not None:
            where_parameters.append(end_date)
            sql_string += " AND %s >= value.datetime "

        sql_string += " ORDER BY value.datetime "
        with connection.cursor() as cursor:

            cursor.execute(sql_string, where_parameters)
            rows = cursor.fetchall()

            for row in rows:
                obj = {
                    'datetime': row[0],
                    'measured': row[1],
                    'consisted': row[2],
                    'automatic_flag': row[3],
                    'manual_flag': row[4],
                    'station_id': row[5],
                    'variable_id': row[6],
                    'remarks': row[7],
                    'ml_flag': row[8],
                }

                response['results'].append(obj)

        return JsonResponse(response)

    if request.method == 'PATCH':
        station_id = request.GET.get('station_id', None)
        variable_id = request.GET.get('variable_id', None)
        req_datetime = request.GET.get('datetime', None)

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id is None:
            JsonResponse(data={"message": "'variable_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        if req_datetime is None:
            JsonResponse(data={"message": "'datetime' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            station_id = int(station_id)
            variable_id = int(variable_id)
            req_datetime = datetime.datetime.strptime(req_datetime, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            JsonResponse(data={"message": "Invalid parameter type."}, status=status.HTTP_400_BAD_REQUEST)

        body = json.loads(request.body.decode('utf-8'))
        query_parameters = []
        sql_columns_to_update = []

        if 'manual_flag' in body:
            try:
                query_parameters.append(parse_int_value(body['manual_flag']))
                sql_columns_to_update.append(' manual_flag=%s ')
            except ValueError:
                return JsonResponse({'message': 'Wrong manual flag value type.'}, status=status.HTTP_400_BAD_REQUEST)

        if 'consisted' in body:
            try:
                query_parameters.append(parse_float_value(body['consisted']))
                sql_columns_to_update.append(' consisted=%s ')
            except ValueError:
                return JsonResponse({'message': 'Wrong consisted value type. Please inform a float value.'},
                                    status=status.HTTP_400_BAD_REQUEST)

        if 'remarks' in body:
            try:
                query_parameters.append(body['remarks'])
                sql_columns_to_update.append(' remarks=%s ')
            except ValueError:
                return JsonResponse({'message': 'Wrong remarks value type. Please inform a text value.'},
                                    status=status.HTTP_400_BAD_REQUEST)

        if not sql_columns_to_update:
            JsonResponse(data={"message": "You must send 'manual_flag', 'consisted' or 'remarks' data to update."},
                         status=status.HTTP_400_BAD_REQUEST)

        if query_parameters:
            query_parameters.append(req_datetime)
            query_parameters.append(station_id)
            query_parameters.append(variable_id)
            sql_query = f"UPDATE raw_data SET {', '.join(sql_columns_to_update)} WHERE datetime=%s AND station_id=%s AND variable_id=%s"

            station = Station.objects.get(pk=station_id)
            with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query, query_parameters)

                    now = datetime.datetime.now()
                    cursor.execute("""
                        INSERT INTO wx_hourlysummarytask (station_id, datetime, updated_at, created_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (station_id, req_datetime, now, now))

                    station_timezone = pytz.UTC
                    if station.utc_offset_minutes is not None:
                        station_timezone = pytz.FixedOffset(station.utc_offset_minutes)

                    date = req_datetime.astimezone(station_timezone).date()
                    cursor.execute("""
                        INSERT INTO wx_dailysummarytask (station_id, date, updated_at, created_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (station_id, date, now, now))

                    cursor.execute(sql_query, query_parameters)

                conn.commit()

            return JsonResponse({}, status=status.HTTP_200_OK)
        return JsonResponse({'message': "There is no 'manual_flag' or 'consisted' fields in the json request."},
                            status=status.HTTP_400_BAD_REQUEST)

    return JsonResponse({'message': 'Only the GET and PATCH methods is allowed.'}, status=status.HTTP_400_BAD_REQUEST)


@csrf_exempt
def interpolate_endpoint(request):
    request_date = request.GET.get('request_date', None)
    variable_id = request.GET.get('variable_id', None)
    data_type = request.GET.get('data_type', None)

    sql_string = ''
    where_parameters = [variable_id, request_date]

    if data_type == 'daily':
        sql_string = """
            SELECT station.latitude
                ,station.longitude
                ,daily.avg_value
            FROM daily_summary as daily
            INNER JOIN wx_station as station ON daily.station_id=station.id
            WHERE daily.variable_id=%s and daily.day=%s
            ORDER BY daily.day
        """

    elif data_type == 'hourly':

        sql_string = """
            SELECT station.latitude
                  ,station.longitude
                  ,hourly.avg_value
            FROM hourly_summary as hourly
            INNER JOIN wx_station as station ON hourly.station_id=station.id
            WHERE hourly.variable_id=%s and hourly.datetime=%s
            ORDER BY hourly.datetime
        """

    elif data_type == 'monthly':
        sql_string = """
            SELECT station.latitude
                  ,station.longitude
                  ,monthly.avg_value
            FROM monthly_summary as monthly
            INNER JOIN wx_station as station ON monthly.station_id=station.id
            WHERE monthly.variable_id=%s and to_date(concat(lpad(monthly.year::varchar(4),4,'0'), lpad(monthly.month::varchar(2),2,'0'), '01'), 'YYYYMMDD')=date_trunc('month',TIMESTAMP %s)
            ORDER BY monthly.year, monthly.month
        """

    if sql_string and where_parameters:
        query_result = []
        with connection.cursor() as cursor:
            cursor.execute(sql_string, where_parameters)
            rows = cursor.fetchall()

            for row in rows:
                obj = {
                    'station__latitude': row[0],
                    'station__longitude': row[1],
                    'value': row[2],
                }

                query_result.append(obj)

            if not query_result:
                return JsonResponse({'message': 'No data found.'}, status=status.HTTP_400_BAD_REQUEST)

            return HttpResponse(get_interpolation_image(query_result), content_type="image/jpeg")

    return JsonResponse({'message': 'Missing parameters.'}, status=status.HTTP_400_BAD_REQUEST)


@csrf_exempt
def capture_forms_values_get(request):
    request_month = request.GET.get('request_month', None)
    station_id = request.GET.get('station_id', None)
    variable_id = request.GET.get('variable_id', None)

    sql_string = """
        select to_char(date.day, 'DD')
              ,to_char(date.day, 'HHAM')
              ,values.measured
              ,%s
              ,%s
              ,date.day
        from ( select generate_series(to_date(%s, 'YYYY-MM') + interval '6 hours', (to_date(%s, 'YYYY-MM') + interval '1 month -1 second'), '12 hours') as day ) date
        LEFT JOIN raw_data as values ON date.day=values.datetime and values.station_id=%s and values.variable_id=%s
        ORDER BY date.day
    """
    where_parameters = [station_id, variable_id, request_month, request_month, station_id, variable_id]

    with connection.cursor() as cursor:
        cursor.execute(sql_string, where_parameters)
        rows = cursor.fetchall()

        days = rows

        days = {}
        for row in rows:
            obj = {
                'value': row[2],
                'station_id': row[3],
                'variable_id': row[4],
                'datetime': row[5],
            }

            if row[0] not in days.keys():
                days[row[0]] = {}

            days[row[0]][row[1]] = obj

        full_list = []
        for day in days.keys():
            line = {}
            for obj in days[day]:
                line[obj] = days[day][obj]
            full_list.append(line)

        if not days:
            return JsonResponse({'message': 'No data found.'}, status=status.HTTP_400_BAD_REQUEST)

        return JsonResponse({'next': None, 'results': full_list}, safe=False, status=status.HTTP_200_OK)

    return JsonResponse({'message': 'Missing parameters.'}, status=status.HTTP_400_BAD_REQUEST)


'''
@csrf_exempt
def capture_forms_values_patch(request):
    if request.method == 'PATCH':

        body = json.loads(request.body.decode('utf-8'))

        conn = psycopg2.connect(settings.SURFACE_CONNECTION_STRING)
        with conn.cursor() as cursor:
            cursor.executemany(
               """
                    INSERT INTO raw_data(datetime, station_id, variable_id, measured) VALUES(%(datetime)s, %(station_id)s, %(variable_id)s, %(value)s::double precision)
                    ON CONFLICT (datetime, station_id, variable_id)
                    DO UPDATE
                    SET measured = %(value)s
                """, body)

        conn.commit()
        conn.close()

        return JsonResponse({}, status=status.HTTP_200_OK)
    return JsonResponse({'message':'Only the GET and PATCH methods is allowed.'}, status=status.HTTP_400_BAD_REQUEST)
'''


@csrf_exempt
def capture_forms_values_patch(request):
    error_flag = False
    if request.method == 'PATCH':

        body = json.loads(request.body.decode('utf-8'))

        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                for rec in body:
                    try:
                        if not rec['value']:
                            cursor.execute(
                                """ DELETE FROM raw_data WHERE datetime=%(datetime)s and station_id=%(station_id)s and variable_id=%(variable_id)s """,
                                rec)
                        else:
                            valor = float(rec['value'])

                            cursor.execute(
                                """
                                    INSERT INTO raw_data(datetime, station_id, variable_id, measured) VALUES(%(datetime)s, %(station_id)s, %(variable_id)s, %(value)s::double precision)
                                    ON CONFLICT (datetime, station_id, variable_id)
                                    DO UPDATE
                                    SET measured = %(value)s
                                """, rec)
                    except (ValueError, psycopg2.errors.InvalidTextRepresentation):
                        error_flag = True

            conn.commit()

        if error_flag:
            return JsonResponse({'message': 'Some data was bad formated, please certify that the input is numeric.'},
                                status=status.HTTP_200_OK)

        return JsonResponse({}, status=status.HTTP_200_OK)
    return JsonResponse({'message': 'Only the GET and PATCH methods is allowed.'}, status=status.HTTP_400_BAD_REQUEST)


class StationListView(LoginRequiredMixin, ListView):
    model = Station


class StationDetailView(LoginRequiredMixin, DetailView):
    model = Station


class StationCreate(LoginRequiredMixin, SuccessMessageMixin, CreateView):
    model = Station
    success_message = "%(name)s was created successfully"
    form_class = StationForm

    layout = Layout(
        Fieldset('Registering a new station',
                 Row('name'),
                 Row('is_active', 'is_automatic'),
                 Row('alias_name'),
                 Row('code', 'profile'),
                 Row('wmo', 'organization'),
                 Row('wigos', 'observer'),
                 Row('begin_date', 'data_source'),
                 Row('end_date')
                 ),
        Fieldset('Other information',
                 Row('latitude', 'longitude'),
                 Row('elevation', 'watershed'),
                 Row('country', 'region'),
                 Row('utc_offset_minutes', 'local_land_use'),
                 Row('soil_type', 'station_details'),
                 Row('site_description', 'alternative_names')
                 ),
        Fieldset('Hydrology information',
                 Row('hydrology_station_type', 'ground_water_province'),
                 Row('existing_gauges', 'flow_direction_at_station'),
                 Row('flow_direction_above_station', 'flow_direction_below_station'),
                 Row('bank_full_stage', 'bridge_level'),
                 Row('temporary_benchmark', 'mean_sea_level'),
                 Row('river_code', 'river_course'),
                 Row('catchment_area_station', 'river_origin'),
                 Row('easting', 'northing'),
                 Row('river_outlet', 'river_length'),
                 Row('z', 'land_surface_elevation'),
                 Row('top_casing_land_surface', 'casing_diameter'),
                 Row('screen_length', 'depth_midpoint'),
                 Row('casing_type', 'datum'),
                 Row('zone')
                 )
    )


class StationUpdate(LoginRequiredMixin, SuccessMessageMixin, UpdateView):
    model = Station
    success_message = "%(name)s was updated successfully"
    form_class = StationForm

    layout = Layout(
        Fieldset('Editing station',
                 Row('name', 'is_active'),
                 Row('alias_name', 'is_automatic'),
                 Row('code', 'profile'),
                 Row('wmo', 'organization'),
                 Row('wigos', 'observer'),
                 Row('begin_date', 'data_source'),
                 Row('end_date', 'communication_type')
                 ),
        Fieldset('Other information',
                 Row('latitude', 'longitude'),
                 Row('elevation', 'watershed'),
                 Row('country', 'region'),
                 Row('utc_offset_minutes', 'local_land_use'),
                 Row('soil_type', 'station_details'),
                 Row('site_description', 'alternative_names')
                 ),
        Fieldset('Hydrology information',
                 Row('hydrology_station_type', 'ground_water_province'),
                 Row('existing_gauges', 'flow_direction_at_station'),
                 Row('flow_direction_above_station', 'flow_direction_below_station'),
                 Row('bank_full_stage', 'bridge_level'),
                 Row('temporary_benchmark', 'mean_sea_level'),
                 Row('river_code', 'river_course'),
                 Row('catchment_area_station', 'river_origin'),
                 Row('easting', 'northing'),
                 Row('river_outlet', 'river_length'),
                 Row('z', 'land_surface_elevation'),
                 Row('top_casing_land_surface', 'casing_diameter'),
                 Row('screen_length', 'depth_midpoint'),
                 Row('casing_type', 'datum'),
                 Row('zone')
                 )

    )


@api_view(['POST'])
def pgia_update(request):
    try:
        hours_dict = request.data['table']
        now_utc = datetime.datetime.now().astimezone(pytz.UTC) + datetime.timedelta(
            hours=settings.PGIA_REPORT_HOURS_AHEAD_TIME)

        pgia = Station.objects.get(id=4)
        datetime_offset = pytz.FixedOffset(pgia.utc_offset_minutes)

        day = datetime.datetime.strptime(request.data['date'], '%Y-%m-%d')
        station_id = pgia.id
        seconds = 3600

        records_list = []
        for hour, hour_data in hours_dict.items():
            data_datetime = day.replace(hour=int(hour))
            data_datetime = datetime_offset.localize(data_datetime)
            if data_datetime <= now_utc:
                if hour_data:
                    if 'action' in hour_data.keys():
                        hour_data.pop('action')

                    if 'remarks' in hour_data.keys():
                        remarks = hour_data.pop('remarks')
                    else:
                        remarks = None

                    if 'observer' in hour_data.keys():
                        observer = hour_data.pop('observer')
                    else:
                        observer = None

                    for variable_id, measurement in hour_data.items():
                        if measurement is None:
                            measurement_value = settings.MISSING_VALUE
                            measurement_code = settings.MISSING_VALUE_CODE

                        try:
                            measurement_value = float(measurement)
                            measurement_code = measurement
                        except Exception:
                            measurement_value = settings.MISSING_VALUE
                            measurement_code = settings.MISSING_VALUE_CODE
                        records_list.append((
                            station_id, variable_id, seconds, data_datetime, measurement_value, 1, None,
                            None, None, None, None, None, None, None, False, remarks, observer,
                            measurement_code))

        insert_raw_data_pgia.insert(raw_data_list=records_list, date=day, station_id=station_id,
                                    override_data_on_conflict=True, utc_offset_minutes=pgia.utc_offset_minutes)
    except Exception as e:
        logger.error(repr(e))
        return HttpResponse(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return HttpResponse(status=status.HTTP_200_OK)


@api_view(['GET'])
def pgia_load(request):
    try:
        date = datetime.datetime.strptime(request.GET['date'], '%Y-%m-%d')
    except ValueError as e:
        logger.error(repr(e))
        return HttpResponse(status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(repr(e))
        return HttpResponse(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    pgia = Station.objects.get(id=4)
    datetime_offset = pytz.FixedOffset(pgia.utc_offset_minutes)
    request_datetime = datetime_offset.localize(date)

    start_datetime = request_datetime
    end_datetime = request_datetime + datetime.timedelta(days=1)

    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                    SELECT (datetime + interval '{pgia.utc_offset_minutes} minutes') at time zone 'utc',
                        variable_id,
                        CASE WHEN var.variable_type ilike 'code' THEN code ELSE measured::varchar END as value,
                        remarks,
                        observer
                    FROM raw_data
                    JOIN wx_variable var ON raw_data.variable_id=var.id
                    WHERE station_id = %(station_id)s
                      AND datetime >= %(start_date)s 
                      AND datetime < %(end_date)s
                """,
                {
                    'start_date': start_datetime,
                    'end_date': end_datetime,
                    'station_id': pgia.id
                })

            response = cursor.fetchall()

    return JsonResponse(response, status=status.HTTP_200_OK, safe=False)


@api_view(['GET'])
def MonthlyFormLoad(request):
    try:
        start_date = datetime.datetime.strptime(request.GET['date'], '%Y-%m')
        station_id = int(request.GET['station'])
    except ValueError as e:
        return JsonResponse({}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return JsonResponse({}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    station = Station.objects.get(id=station_id)
    datetime_offset = pytz.FixedOffset(station.utc_offset_minutes)
    end_date = start_date.replace(month=start_date.month + 1) - datetime.timedelta(days=1)

    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                    SELECT (datetime + interval '{station.utc_offset_minutes} minutes') at time zone 'utc'
                          ,variable_id
                          ,measured
                    FROM raw_data
                    WHERE station_id = %(station_id)s
                      AND datetime >= %(start_date)s 
                      AND datetime <= %(end_date)s
                """,
                {
                    'start_date': start_date,
                    'end_date': end_date,
                    'station_id': station_id
                })

            response = cursor.fetchall()

        conn.commit()

    return JsonResponse(response, status=status.HTTP_200_OK, safe=False)


@api_view(['POST'])
def MonthlyFormUpdate(request):
    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        records_list = []

        station_id = int(request.data['station'])
        station = Station.objects.get(id=station_id)
        first_day = datetime.datetime(year=int(request.data['date']['year']), month=int(request.data['date']['month']),
                                      day=1)

        now_utc = datetime.datetime.now().astimezone(pytz.UTC)
        datetime_offset = pytz.FixedOffset(station.utc_offset_minutes)

        days_in_month = (first_day.replace(month=first_day.month + 1) - datetime.timedelta(days=1)).day

        for day in range(0, days_in_month):
            data = request.data['table'][day]
            data_datetime = first_day.replace(day=day + 1)
            data_datetime = datetime_offset.localize(data_datetime)

            if data_datetime <= now_utc:
                for variable_id, value in data.items():
                    if value is None:
                        value = settings.MISSING_VALUE

                    records_list.append((
                        station_id, variable_id, 86400, data_datetime, value, 1, None, None, None, None,
                        None, None, None, None, False, None, None, None))

    insert_raw_data_pgia.insert(raw_data_list=records_list, date=first_day, station_id=station_id,
                                override_data_on_conflict=True, utc_offset_minutes=station.utc_offset_minutes)
    return JsonResponse({}, status=status.HTTP_200_OK)


class StationDelete(LoginRequiredMixin, DeleteView):
    model = Station
    fields = ['code', 'name', 'profile', ]

    def get_success_url(self):
        return reverse('stations-list')


class StationFileList(LoginRequiredMixin, ListView):
    model = StationFile

    def get_queryset(self):
        queryset = StationFile.objects.filter(station__id=self.kwargs.get('pk'))
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        station = Station.objects.get(pk=self.kwargs.get(
            'pk'))  # the self.kwargs is different from **kwargs, and gives access to the named url parameters
        context['station'] = station
        return context


class StationFileCreate(LoginRequiredMixin, SuccessMessageMixin, CreateView):
    model = StationFile
    # fields = "__all__"
    fields = ('name', 'file')
    success_message = "%(name)s was created successfully"
    layout = Layout(
        Fieldset('Add file to station',
                 Row('name'),
                 Row('file')
                 )
    )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        station = Station.objects.get(pk=self.kwargs.get(
            'pk'))  # the self.kwargs is different from **kwargs, and gives access to the named url parameters
        context['station'] = station
        return context

    def form_valid(self, form):
        f = form.save(commit=False)
        station = Station.objects.get(pk=self.kwargs.get('pk'))
        f.station = station
        f.save()
        return super(StationFileCreate, self).form_valid(form)

    def get_success_url(self):
        return reverse('stationfiles-list', kwargs={'pk': self.kwargs.get('pk')})


class StationFileDelete(LoginRequiredMixin, DeleteView):
    model = StationFile
    success_message = "%(name)s was deleted successfully"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        station = Station.objects.get(pk=self.kwargs.get(
            'pk_station'))  # the self.kwargs is different from **kwargs, and gives access to the named url parameters
        context['station'] = station
        return context

    def get_success_url(self):
        return reverse('stationfiles-list', kwargs={'pk': self.kwargs.get('pk_station')})


class StationVariableListView(LoginRequiredMixin, ListView):
    model = StationVariable

    def get_queryset(self):
        queryset = StationVariable.objects.filter(station__id=self.kwargs.get('pk'))
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        station = Station.objects.get(pk=self.kwargs.get(
            'pk'))  # the self.kwargs is different from **kwargs, and gives access to the named url parameters
        context['station'] = station
        return context


class StationVariableCreateView(LoginRequiredMixin, SuccessMessageMixin, CreateView):
    model = StationVariable
    fields = ('variable',)
    success_message = "%(variable)s was created successfully"
    layout = Layout(
        Fieldset('Add variable to station',
                 Row('variable')
                 )
    )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        station = Station.objects.get(pk=self.kwargs.get(
            'pk'))  # the self.kwargs is different from **kwargs, and gives access to the named url parameters
        context['station'] = station
        return context

    def form_valid(self, form):
        f = form.save(commit=False)
        station = Station.objects.get(pk=self.kwargs.get('pk'))
        f.station = station
        f.save()
        return super(StationVariableCreateView, self).form_valid(form)

    def get_success_url(self):
        return reverse('stationvariable-list', kwargs={'pk': self.kwargs.get('pk')})


class StationVariableDeleteView(LoginRequiredMixin, DeleteView):
    model = StationVariable
    success_message = "%(action)s was deleted successfully"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        station = Station.objects.get(pk=self.kwargs.get(
            'pk_station'))  # the self.kwargs is different from **kwargs, and gives access to the named url parameters
        context['station'] = station
        return context

    def get_success_url(self):
        return reverse('stationvariable-list', kwargs={'pk': self.kwargs.get('pk_station')})


def station_report_data(request):
    station = request.GET.get('station', None)
    initial_datetime = request.GET.get('initial_datetime', None)
    final_datetime = request.GET.get('final_datetime', None)
    source = request.GET.get('source', None)

    if station and initial_datetime and final_datetime and source:
        if int(source) == 0:  # Raw data
            dataset = get_raw_data('station', station, None, initial_datetime, final_datetime, 'raw_data')
        elif int(source) == 1:  # Hourly data
            dataset = get_raw_data('station', station, None, initial_datetime, final_datetime, 'hourly_summary')
        elif int(source) == 2:  # Daily data
            dataset = get_raw_data('station', station, None, initial_datetime, final_datetime, 'daily_summary')
        elif int(source) == 3:  # Monthly data
            dataset = get_raw_data('station', station, None, initial_datetime, final_datetime, 'monthly_summary')
        elif int(source) == 4:  # Yearly data
            dataset = get_raw_data('station', station, None, initial_datetime, final_datetime, 'yearly_summary')
        else:
            return JsonResponse({}, status=status.HTTP_404_NOT_FOUND)

        charts = {

        }

        for element_name, element_data in dataset['results'].items():

            chart = {
                'chart': {
                    'type': 'pie',
                    'zoomType': 'xy'
                },
                'title': {'text': element_name},
                'xAxis': {
                    'type': 'datetime',
                    'dateTimeLabelFormats': {
                        'month': '%e. %b',
                        'year': '%b'
                    },
                    'title': {
                        'text': 'Date'
                    }
                },
                'yAxis': [],
                'exporting': {
                    'showTable': True
                },
                'series': []
            }

            opposite = False
            y_axis_unit_dict = {}
            for variable_name, variable_data in element_data.items():

                current_unit = variable_data['unit']
                if current_unit not in y_axis_unit_dict.keys():
                    chart['yAxis'].append({
                        'labels': {
                            'format': '{value} ' + variable_data['unit']
                        },
                        'title': {
                            'text': None
                        },
                        'opposite': opposite
                    })
                    y_axis_unit_dict[current_unit] = len(chart['yAxis']) - 1
                    opposite = not opposite

                current_y_axis_index = y_axis_unit_dict[current_unit]
                data = []
                for record in variable_data['data']:
                    if int(source) == 0:
                        data.append({
                            'x': record['date'],
                            'y': record['value'],
                            'quality_flag': record['quality_flag'],
                            'flag_color': record['flag_color']
                        })
                    elif int(source) == 3:
                        data.append({
                            'x': record['date'],
                            'y': record['value']
                        })

                        chart['xAxis'] = {
                            'type': 'datetime',
                            'labels': {
                                'format': '{value:%Y-%b}'
                            },
                            'title': {
                                'text': 'Y'
                            }
                        }
                    elif int(source) == 4:
                        data.append({
                            'x': record['date'],
                            'y': record['value']
                        })

                        chart['xAxis'] = {
                            'type': 'datetime',
                            'labels': {
                                'format': '{value:%Y-%b}'
                            },
                            'title': {
                                'text': 'Reference'
                            }
                        }
                    else:
                        data.append({
                            'x': record['date'],
                            'y': record['value']
                        })

                chart['series'].append({
                    'name': variable_name,
                    'color': variable_data['color'],
                    'type': variable_data['default_representation'],
                    'unit': variable_data['unit'],
                    'data': data,
                    'yAxis': current_y_axis_index
                })
                chart['chart']['type'] = variable_data['default_representation'],

            charts[slugify(element_name)] = chart

        return JsonResponse(charts)
    else:
        return JsonResponse({}, status=status.HTTP_400_BAD_REQUEST)


def variable_report_data(request):
    variable_ids_list = request.GET.get('variable_ids', None)
    initial_datetime = request.GET.get('initial_datetime', None)
    final_datetime = request.GET.get('final_datetime', None)
    source = request.GET.get('source', None)
    station_id_list = request.GET.get('station_ids', None)

    if variable_ids_list and initial_datetime and final_datetime and source and station_id_list:
        variable_ids_list = tuple(json.loads(variable_ids_list))
        station_id_list = tuple(json.loads(station_id_list))

        if int(source) == 0:  # Raw data
            dataset = get_station_raw_data('variable', variable_ids_list, None, initial_datetime, final_datetime,
                                           station_id_list, 'raw_data')
        elif int(source) == 1:  # Hourly data
            dataset = get_station_raw_data('variable', variable_ids_list, None, initial_datetime, final_datetime,
                                           station_id_list, 'hourly_summary')
        elif int(source) == 2:  # Daily data
            dataset = get_station_raw_data('variable', variable_ids_list, None, initial_datetime, final_datetime,
                                           station_id_list, 'daily_summary')
        elif int(source) == 3:  # Monthly data
            dataset = get_station_raw_data('variable', variable_ids_list, None, initial_datetime, final_datetime,
                                           station_id_list, 'monthly_summary')
        elif int(source) == 4:  # Yearly data
            dataset = get_station_raw_data('variable', variable_ids_list, None, initial_datetime, final_datetime,
                                           station_id_list, 'yearly_summary')
        else:
            return JsonResponse({}, status=status.HTTP_404_NOT_FOUND)

        charts = {

        }

        for element_name, element_data in dataset['results'].items():

            chart = {
                'chart': {
                    'type': 'pie',
                    'zoomType': 'xy'
                },
                'title': {'text': element_name},
                'xAxis': {
                    'type': 'datetime',
                    'dateTimeLabelFormats': {
                        'month': '%e. %b',
                        'year': '%b'
                    },
                    'title': {
                        'text': 'Date'
                    }
                },
                'yAxis': [],
                'exporting': {
                    'showTable': True
                },
                'series': []
            }

            opposite = False
            y_axis_unit_dict = {}
            for variable_name, variable_data in element_data.items():

                current_unit = variable_data['unit']
                if current_unit not in y_axis_unit_dict.keys():
                    chart['yAxis'].append({
                        'labels': {
                            'format': '{value} ' + variable_data['unit']
                        },
                        'title': {
                            'text': None
                        },
                        'opposite': opposite
                    })
                    y_axis_unit_dict[current_unit] = len(chart['yAxis']) - 1
                    opposite = not opposite

                current_y_axis_index = y_axis_unit_dict[current_unit]
                data = []
                for record in variable_data['data']:
                    if int(source) == 0:
                        data.append({
                            'x': record['date'],
                            'y': record['value'],
                            'quality_flag': record['quality_flag'],
                            'flag_color': record['flag_color']
                        })
                    elif int(source) == 3:
                        data.append({
                            'x': record['date'],
                            'y': record['value']
                        })

                        chart['xAxis'] = {
                            'type': 'datetime',
                            'labels': {
                                'format': '{value:%Y-%b}'
                            },
                            'title': {
                                'text': 'Y'
                            }
                        }
                    elif int(source) == 4:
                        data.append({
                            'x': record['date'],
                            'y': record['value']
                        })

                        chart['xAxis'] = {
                            'type': 'datetime',
                            'labels': {
                                'format': '{value:%Y-%b}'
                            },
                            'title': {
                                'text': 'Reference'
                            }
                        }
                    else:
                        data.append({
                            'x': record['date'],
                            'y': record['value']
                        })

                chart['series'].append({
                    'name': variable_name,
                    'color': "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
                    'type': variable_data['default_representation'],
                    'unit': variable_data['unit'],
                    'data': data,
                    'yAxis': current_y_axis_index
                })
                chart['chart']['type'] = variable_data['default_representation'],

            charts[slugify(element_name)] = chart

        return JsonResponse(charts)
    else:
        return JsonResponse({}, status=status.HTTP_400_BAD_REQUEST)


class StationReportView(LoginRequiredMixin, TemplateView):
    template_name = "wx/products/station_report.html"

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_id'] = request.GET.get('station_id', 'null')

        station_list = Station.objects.all()
        context['station_list'] = station_list

        # interval_list = Interval.objects.filter(seconds__lte=3600).order_by('seconds')
        # context['interval_list'] = interval_list

        quality_flag_query = QualityFlag.objects.all()
        quality_flag_colors = {}
        for quality_flag in quality_flag_query:
            quality_flag_colors[quality_flag.name.replace(' ', '_')] = quality_flag.color
        context['quality_flag_colors'] = quality_flag_colors

        selected_station = station_list.first()
        if selected_station is not None:
            selected_station_id = selected_station.id
            station_variable_list = StationVariable.objects.filter(station__id=selected_station_id)
        else:
            station_variable_list = []

        context['station_variable_list'] = station_variable_list

        return self.render_to_response(context)


class VariableReportView(LoginRequiredMixin, TemplateView):
    template_name = "wx/products/variable_report.html"

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)

        quality_flag_query = QualityFlag.objects.all()
        quality_flag_colors = {}
        for quality_flag in quality_flag_query:
            quality_flag_colors[quality_flag.name.replace(' ', '_')] = quality_flag.color

        context['quality_flag_colors'] = quality_flag_colors
        context['variable_list'] = Variable.objects.all()
        context['station_list'] = Station.objects.all()

        return self.render_to_response(context)


class ProductReportView(LoginRequiredMixin, TemplateView):
    template_name = "wx/products/report.html"


class ProductCompareView(LoginRequiredMixin, TemplateView):
    template_name = 'wx/products/compare.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.select_related('profile').all()
        context['variable_list'] = Variable.objects.select_related('unit').all()

        context['station_profile_list'] = StationProfile.objects.all()
        context['station_watershed_list'] = Watershed.objects.all()
        context['station_district_list'] = AdministrativeRegion.objects.all()

        return self.render_to_response(context)


class QualityControlView(LoginRequiredMixin, TemplateView):
    template_name = 'wx/quality_control/validation.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.select_related('profile').all()

        context['station_profile_list'] = StationProfile.objects.all()
        context['station_watershed_list'] = Watershed.objects.all()
        context['station_district_list'] = AdministrativeRegion.objects.all()

        return self.render_to_response(context)


@csrf_exempt
def get_yearly_average(request):
    begin_date = request.GET.get('begin_date', None)
    end_date = request.GET.get('end_date', None)
    station_id = request.GET.get('station_id', None)
    variable_id = request.GET.get('variable_id', None)

    sql_string = """
        select avg(val.measured) as average, extract ('year' from month)::varchar as year, extract ('month' from month)::varchar as month, round(avg(val.measured)::decimal, 2)
        from generate_series(date_trunc('year', date %s),date_trunc('year', date %s) + interval '1 year' - interval '1 day', interval '1 month')  month
        left outer join raw_data as val on date_trunc('month', val.datetime) = date_trunc('month', month) and val.station_id = %s and val.variable_id = %s
        group by month
        order by month
    """
    where_parameters = [begin_date, end_date, station_id, variable_id]

    with connection.cursor() as cursor:
        cursor.execute(sql_string, where_parameters)
        rows = cursor.fetchall()

        years = {}
        for row in rows:
            if row[1] not in years.keys():
                years[row[1]] = {}

            if row[2] not in years[row[1]].keys():
                years[row[1]][row[2]] = row[3]

            years[row[1]][row[2]] = row[3]

        if not years:
            return JsonResponse({'message': 'No data found.'}, status=status.HTTP_400_BAD_REQUEST)

        return JsonResponse({'next': None, 'results': years}, safe=False, status=status.HTTP_200_OK)

    return JsonResponse({'message': 'Missing parameters.'}, status=status.HTTP_400_BAD_REQUEST)


class YearlyAverageReport(LoginRequiredMixin, TemplateView):
    template_name = 'wx/reports/yearly_average.html'


class StationVariableStationViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = StationVariable.objects.values('station__id', 'station__name', 'station__code').distinct('station__id')
    serializer_class = serializers.ReducedStationSerializer

    def get_queryset(self):
        queryset = StationVariable.objects.values('station__id', 'station__name', 'station__code').distinct(
            'station__id')

        variable_id = self.request.query_params.get('variable_id', None)

        if variable_id is not None:
            queryset = queryset.filter(variable__id=variable_id)

        return queryset


def last24_summary_list(request):
    search_type = request.GET.get('search_type', None)
    search_value = request.GET.get('search_value', None)

    response = {
        'results': [],
        'messages': [],
    }

    if search_type is not None and search_type == 'variable':
        query = """
            SELECT last24h.datetime,
                   var.name,
                   var.symbol,
                   var.sampling_operation_id,
                   unit.name,
                   unit.symbol,
                   last24h.min_value,
                   last24h.max_value,
                   last24h.avg_value,
                   last24h.sum_value,
                   last24h.latest_value,
                   last24h.station_id,
                   last24h.num_records
            FROM last24h_summary last24h
            JOIN wx_variable var ON last24h.variable_id=var.id
            JOIN wx_unit unit ON var.unit_id=unit.id
            WHERE last24h.variable_id=%s
        """

    with connection.cursor() as cursor:

        cursor.execute(query, [search_value])
        rows = cursor.fetchall()

        for row in rows:

            value = None

            if row[3] == 1:
                value = row[10]

            elif row[3] == 2:
                value = row[8]

            elif row[3] == 3:
                value = row[6]

            elif row[3] == 4:
                value = row[7]

            elif row[3] == 6:
                value = row[9]

            if value is None:
                print('variable {} does not have supported sampling operation {}'.format(row[1], row[3]))

            obj = {
                'station': row[11],
                'value': value,
                'min': round(row[6], 2),
                'max': round(row[7], 2),
                'avg': round(row[8], 2),
                'sum': round(row[9], 2),
                'count': row[12],
                'variable': {
                    'name': row[1],
                    'symbol': row[2],
                    'unit_name': row[4],
                    'unit_symbol': row[5]
                }
            }
            response['results'].append(obj)

        if response['results']:
            return JsonResponse(response, status=status.HTTP_200_OK)

    return JsonResponse(data={"message": "No data found."}, status=status.HTTP_404_NOT_FOUND)


class ComingSoonView(LoginRequiredMixin, TemplateView):
    template_name = "coming-soon.html"

#################################################################################
#################################################################################

from django.views.decorators.http import require_http_methods
from wx.models import MaintenanceReport, MaintenanceReportStationComponent, StationProfileComponent, StationComponent
from base64 import b64encode

####################################

@require_http_methods(["GET"])
def get_maintenance_reports(request): # 1
    template = loader.get_template('wx/maintenance_reports/maintenance_reports.html')
    context = {}
    return HttpResponse(template.render(context, request))

def get_maintenance_report_date(maintenance_report):
    station = Station.objects.get(id=maintenance_report.station_id)
    station_profile = StationProfile.objects.get(id=station.profile_id)
    technician = Technician.objects.get(id=maintenance_report.responsible_technician_id)
    visit_type = VisitType.objects.get(id=maintenance_report.visit_type_id)

    return station, station_profile, technician, visit_type

@require_http_methods(["PUT"])
def get_maintenance_report_list(request):
    form_data = json.loads(request.body.decode())

    maintenance_reports = MaintenanceReport.objects.filter(visit_date__gte = form_data['start_date'], visit_date__lte = form_data['end_date'])

    response = {}
    response['maintenance_report_list'] = []

    for maintenance_report in maintenance_reports:
        station, station_profile, technician, visit_type = get_maintenance_report_date(maintenance_report)

        if station.is_automatic == form_data['is_automatic']:
            if maintenance_report.status == 'A':
                maintenance_report_status = 'Approved'
            elif maintenance_report.status == 'P':
                maintenance_report_status = 'Published'
            else:
                maintenance_report_status = 'Draft'

            maintenance_report_object = {
                'maintenance_report_id': maintenance_report.id,
                'station_name': station.name,
                'station_profile': station_profile.name,
                'station_type': 'Automatic' if station.is_automatic else 'Manual',
                'visit_date': maintenance_report.visit_date,
                'next_visit_date': maintenance_report.next_visit_date,
                'technician': technician.name,
                'type_of_visit': visit_type.name,
                'status': maintenance_report_status,
            }
            response['maintenance_report_list'].append(maintenance_report_object)

    return JsonResponse(response, status=status.HTTP_200_OK)

####################################

@require_http_methods(["GET"])
def get_maintenance_report_form(request):
    template = loader.get_template('wx/maintenance_reports/new_report.html')

    context = {}
    context['station_list'] = Station.objects.select_related('profile').all()
    context['visittype_list'] = VisitType.objects.all()
    context['technician_list'] = Technician.objects.all()

    # JSON
    # return JsonResponse(context, status=status.HTTP_200_OK)

    return HttpResponse(template.render(context, request))

def get_station_contacts(station_id):
    maintenance_report_list = MaintenanceReport.objects.filter(station_id=station_id).order_by('visit_date')

    for maintenance_report in maintenance_report_list:
        print(maintenance_report.visit_date)

        if maintenance_report.contacts != '':
            return maintenance_report.contacts

    return None

# https://docs.djangoproject.com/en/4.1/topics/http/decorators/
@require_http_methods(["POST"])
def create_maintenance_report(request):
    now = datetime.datetime.now()

    form_data = json.loads(request.body.decode())

    try:
        maintenance_report = MaintenanceReport.objects.get(station_id = form_data['station_id'], visit_date = form_data['visit_date'])
        

        response={"maintenance_report_id": maintenance_report.id}
        return JsonResponse(response, status=status.HTTP_200_OK)

        response = {"message": "Maintenance report already exist for chosen station and date."}        
        return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)

    except ObjectDoesNotExist:
        maintenance_report = MaintenanceReport.objects.create(
                                    created_at = now,
                                    updated_at = now,
                                    station_id = form_data['station_id'],
                                    responsible_technician_id = form_data['responsible_technician_id'],
                                    visit_type_id = form_data['visittype_id'],
                                    visit_date = form_data['visit_date'],
                                    initial_time = form_data['initial_time'],
                                    contacts = get_station_contacts(form_data['station_id']),
                            )

        station = Station.objects.get(pk=maintenance_report.station_id)
        station_profile_component_list = StationProfileComponent.objects.filter(profile_id = station.profile_id)

        for station_profile_component in station_profile_component_list:
            station_component = StationComponent.objects.get(id=station_profile_component.station_component_id)

            maintenance_report_station_component = MaintenanceReportStationComponent.objects.create(
                                                        maintenance_report_id = maintenance_report.id,
                                                        station_component_id = station_component.id,
                                                        condition = station_component.report_template,
                                                   )

        response={"maintenance_report_id": maintenance_report.id}

        return JsonResponse(response, status=status.HTTP_200_OK)

@require_http_methods(["GET"])
def get_maintenance_report(request, id):
    maintenance_report = MaintenanceReport.objects.get(id=id)

    station = Station.objects.get(pk=maintenance_report.station_id)
    responsible_technician = Technician.objects.get(pk=maintenance_report.responsible_technician_id)

    if station.profile_id is not None:
        profile = StationProfile.objects.get(pk=station.profile_id)
        station_profile_component_list = StationProfileComponent.objects.filter(profile_id=station.profile_id)

        maintenance_report_station_component_list = []
        
        for station_profile_component in station_profile_component_list:
            station_component = StationComponent.objects.get(id=station_profile_component.station_component_id)
            maintenance_report_station_component = MaintenanceReportStationComponent.objects.get(maintenance_report_id=maintenance_report.id,
                                                                                                 station_component_id=station_component.id)

            dictionary = {'component_id': station_component.id,
                          'presentation_order': station_profile_component.presentation_order,
                          'component_name': station_component.name,
                          'condition': maintenance_report_station_component.condition,
                          'component_classification': maintenance_report_station_component.component_classification}

            maintenance_report_station_component_list.append(dictionary)

        maintenance_report_station_component_list = sorted(maintenance_report_station_component_list, key=lambda d: d['presentation_order']) 

    response = {}
    response['responsible_technician'] = {
        "name": responsible_technician.name,
    }
    response['station'] = {
        "name": station.name,
        "station_type": 'Automatic' if station.is_automatic else 'Manual',
        "elevation": station.elevation,
        "data_of_first_operation": station.begin_date,
        "latitude": station.latitude,
        "district": station.region,
        "data_of_relocation": station.relocation_date,
        "code": station.code,
        "longitude": station.longitude,
        "watershed": station.watershed,
        "wigos": station.wigos,
        "profile": profile.name,
        # "transmission_type": station.latitude,            # Review
        # "host_name": station.alias_name,                  # Review
        # "transmission_id": variable_symbol_dict,          # Review
        # "transmission_interval": variable_symbol_dict,    # Review
        # "measurement_interval": variable_symbol_dict,     # Review
    }

    response['station_on_arrival_conditions'] = maintenance_report.station_on_arrival_conditions

    response['contacts'] = maintenance_report.contacts

    response['station_on_arrival_conditions'] = maintenance_report.station_on_arrival_conditions


    response['other_technician_1'] = maintenance_report.other_technician_1_id
    response['other_technician_2'] = maintenance_report.other_technician_2_id
    response['other_technician_3'] = maintenance_report.other_technician_3_id
    response['next_visit_date'] = maintenance_report.next_visit_date
    response['end_time'] = maintenance_report.end_time
    response['current_visit_summary'] = maintenance_report.current_visit_summary
    response['next_visit_summary'] = maintenance_report.next_visit_summary

    # Needs sorting by presentation order
    response['component_list'] = maintenance_report_station_component_list

    response['steps'] = len(station_profile_component_list)

    return JsonResponse(response, status=status.HTTP_200_OK)

@require_http_methods(["PUT"])
def update_maintenance_report_condition(request, id):
    now = datetime.datetime.now()

    maintenance_report = MaintenanceReport.objects.get(id=id)
    
    form_data = json.loads(request.body.decode())
    
    maintenance_report.station_on_arrival_conditions = form_data['conditions']

    maintenance_report.updated_at = now
    maintenance_report.save()

    response={}

    return JsonResponse(response, status=status.HTTP_200_OK)

@require_http_methods(["PUT"])
def update_maintenance_report_component(request, id, component_id):
    now = datetime.datetime.now()

    maintenance_report_station_component = MaintenanceReportStationComponent.objects.get(maintenance_report_id=id, station_component_id=component_id)

    form_data = json.loads(request.body.decode())

    maintenance_report_station_component.component_classification=form_data['component_classification']
    maintenance_report_station_component.condition=form_data['component_condition']

    maintenance_report_station_component.updated_at = now
    maintenance_report_station_component.save()

    response={}

    return JsonResponse(response, status=status.HTTP_200_OK)

@require_http_methods(["PUT"])
def update_maintenance_report_contacts(request, id):
    now = datetime.datetime.now()

    maintenance_report = MaintenanceReport.objects.get(id=id)

    form_data = json.loads(request.body.decode())

    maintenance_report.contacts = form_data['contacts']

    maintenance_report.updated_at = now
    maintenance_report.save()

    response={}

    return JsonResponse(response, status=status.HTTP_200_OK)

@require_http_methods(["POST"])
def update_maintenance_report_datalogger(request, id):
    # print(request.FILES)
    if 'data_logger_file' in request.FILES:
        print('something')
        now = datetime.datetime.now()
        maintenance_report = MaintenanceReport.objects.get(id=id)

        data_logger_file = request.FILES['data_logger_file'].file
        data_logger_file_content = b64encode(data_logger_file.read()).decode('utf-8')

        maintenance_report.data_logger_file = data_logger_file_content

        maintenance_report.updated_at = now
        maintenance_report.save()

        response={}

        return JsonResponse(response, status=status.HTTP_200_OK)

    # print("Data logger file not uploaded.")
    response={'message': "Data logger file not uploaded."}
    return JsonResponse(response, status=status.HTTP_206_PARTIAL_CONTENT)

@require_http_methods(["PUT"])
def update_maintenance_report_summary(request, id):
    now = datetime.datetime.now()

    maintenance_report = MaintenanceReport.objects.get(id=id)

    form_data = json.loads(request.body.decode())

    if form_data['other_technician_1']:
        other_technician_1 = Technician.objects.get(id=form_data['other_technician_1'])
    else:
        other_technician_1 = None

    if form_data['other_technician_2']:
        other_technician_2 = Technician.objects.get(id=form_data['other_technician_2'])
    else:
        other_technician_2 = None

    if form_data['other_technician_3']:
        other_technician_3 = Technician.objects.get(id=form_data['other_technician_3'])    
    else:
        other_technician_3 = None

    maintenance_report.other_technician_1 = other_technician_1
    maintenance_report.other_technician_2 = other_technician_2
    maintenance_report.other_technician_3 = other_technician_3
    maintenance_report.next_visit_date = form_data['next_visit_date']
    maintenance_report.end_time = form_data['end_time']
    maintenance_report.current_visit_summary = form_data['current_visit_summary']
    maintenance_report.next_visit_summary = form_data['next_visit_summary']
    maintenance_report.status = form_data['status']

    maintenance_report.updated_at = now
    maintenance_report.save()

    response={}

    return JsonResponse(response, status=status.HTTP_200_OK)

@require_http_methods(["PUT"])
def update_maintenance_report(id):
    something=3
    # Delete

####################################

@require_http_methods(["DELTE"])
def delete_maintenance_report(id):
    something=3
    # Delete

####################################

@require_http_methods(["GET"])
def get_maintenance_report_view(request, id):
    template = loader.get_template('wx/maintenance_reports/view_report.html')

    maintenance_report = MaintenanceReport.objects.get(id=id)

    station = Station.objects.get(pk=maintenance_report.station_id)
    profile = StationProfile.objects.get(pk=station.profile_id)
    responsible_technician = Technician.objects.get(pk=maintenance_report.responsible_technician_id)
    visit_type = VisitType.objects.get(pk=maintenance_report.visit_type_id)
    maintenance_report_station_components = MaintenanceReportStationComponent.objects.filter(maintenance_report_id=maintenance_report.id)

    maintenance_report_station_component_list = []    
    for maintenance_report_station_component in maintenance_report_station_components:
        dictionary = {'condition': maintenance_report_station_component.condition,
                      'component_classification': maintenance_report_station_component.component_classification,
                     }
        maintenance_report_station_component_list.append(dictionary)


    other_technicians_ids = [maintenance_report.other_technician_1_id,
                             maintenance_report.other_technician_2_id,
                             maintenance_report.other_technician_3_id]
    other_technicians = []
    for other_technician_id in other_technicians_ids:
        if other_technician_id:
            other_technician = Technician.objects.get(id=other_technician_id)
            other_technicians.append(other_technician.name)

    other_technicians = ", ".join(other_technicians)

    context = {}
    context['visit_summary_information'] = {
        "report_number": maintenance_report.id,
        "responsible_technician": responsible_technician.name,
        "date_of_visit": maintenance_report.visit_date,
        "date_of_next_visit": maintenance_report.next_visit_date,
        "start_time": maintenance_report.initial_time,
        "other_technicians": other_technicians,
        "end_time": maintenance_report.end_time,
        "type_of_visit": visit_type.name,
        "station_on_arrival_conditions": maintenance_report.station_on_arrival_conditions,
        "current_visit_summary": maintenance_report.current_visit_summary,
        "next_visit_summary": maintenance_report.next_visit_summary,
    }

    context['station_information'] = {
        "station_name": station.name,
        "station_host_name": "---",
        "station_ID": station.code,
        "wigos_ID": station.wigos,
        "station_type": 'Automatic' if station.is_automatic else 'Manual',
        "station_profile": profile.name,
        "latitude": station.latitude,
        "elevation": station.elevation,
        "longitude": station.longitude,
        "district": station.region,
        "transmission_type": "---",
        "transmission_ID": "---",
        "transmission_interval": "---",
        "measurement_interval": "---",
        "data_of_first_operation": station.begin_date,
        "data_of_relocation": station.relocation_date,
    }

    context['contact_information'] = maintenance_report.contacts  

    context['equipment_records'] = maintenance_report_station_component_list

    # JSON
    # return JsonResponse(context, status=status.HTTP_200_OK)

    return HttpResponse(template.render(context, request))

# https://docs.djangoproject.com/en/4.1/topics/class-based-views/
# https://docs.djangoproject.com/en/4.1/topics/auth/default/#django.contrib.auth.decorators.login_required

#################################################################################
#################################################################################

class SpatialAnalysisView(LoginRequiredMixin, TemplateView):
    template_name = "wx/spatial_analysis.html"

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['quality_flags'] = QualityFlag.objects.all()

        return self.render_to_response(context)


@api_view(['GET'])
def raw_data_last_24h(request, station_id):
    station = Station.objects.get(pk=station_id)
    response_dict = {}

    query = f"""
        WITH data AS (
            SELECT to_char((datetime + interval '{station.utc_offset_minutes} minutes') at time zone 'utc', 'YYYY-MM-DD HH24:MI:SS') as datetime,
                   measured,
                   variable_id
            FROM raw_data
            WHERE station_id=%s
              AND datetime >= now() - '1 day'::interval
              AND measured != {settings.MISSING_VALUE})
        SELECT datetime,
               var.name,
               var.symbol,
               unit.name,
               unit.symbol,
               measured
        FROM data
            INNER JOIN wx_variable var ON variable_id=var.id
            LEFT JOIN wx_unit unit ON var.unit_id=unit.id
        ORDER BY datetime, var.name
    """

    with connection.cursor() as cursor:

        cursor.execute(query, [station.id])

        rows = cursor.fetchall()

        for row in rows:

            if row[0] not in response_dict.keys():
                response_dict[row[0]] = []

            obj = {
                'value': row[5],
                'variable': {
                    'name': row[1],
                    'symbol': row[2],
                    'unit_name': row[3],
                    'unit_symbol': row[4]
                }
            }
            response_dict[row[0]].append(obj)

    return JsonResponse(response_dict)


class StationsMapView(LoginRequiredMixin, TemplateView):
    template_name = "wx/station_map.html"


class StationMetadataView(LoginRequiredMixin, TemplateView):
    template_name = "wx/station_metadata.html"

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_id'] = kwargs.get('pk', 'null')
        print('kwargs', context['station_id'])

        wmo_station_type_list = WMOStationType.objects.all()
        context['wmo_station_type_list'] = wmo_station_type_list

        wmo_region_list = WMORegion.objects.all()
        context['wmo_region_list'] = wmo_region_list

        wmo_program_list = WMOProgram.objects.all()
        context['wmo_program_list'] = wmo_program_list

        station_profile_list = StationProfile.objects.all()
        context['station_profile_list'] = station_profile_list

        station_communication_list = StationCommunication.objects.all()
        context['station_communication_list'] = station_communication_list

        return self.render_to_response(context)


@api_view(['GET'])
def latest_data(request, variable_id):
    result = []

    query = """
        SELECT CASE WHEN var.variable_type ilike 'code' THEN latest.last_data_code ELSE latest.last_data_value::varchar END as value,
               sta.longitude,
               sta.latitude,
               unit.symbol
        FROM wx_stationvariable latest
        INNER JOIN wx_variable var ON latest.variable_id=var.id
        INNER JOIN wx_station sta ON latest.station_id=sta.id
        LEFT JOIN wx_unit unit ON var.unit_id=unit.id
        WHERE latest.variable_id=%s 
          AND latest.last_data_value is not null
          AND latest.last_data_datetime = ( SELECT MAX(most_recent.last_data_datetime)
                                            FROM wx_stationvariable most_recent
                                            WHERE most_recent.station_id=latest.station_id 
                                                AND most_recent.last_data_value is not null)
        """

    with connection.cursor() as cursor:
        cursor.execute(query, [variable_id])

        rows = cursor.fetchall()

        for row in rows:
            obj = {
                'value': row[0],
                'longitude': row[1],
                'latitude': row[2],
                'symbol': row[3],
            }

            result.append(obj)

    return Response(result, status=status.HTTP_200_OK)


class StationImageViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    parser_class = (FileUploadParser,)
    queryset = StationImage.objects.all()
    serializer_class = serializers.StationImageSerializer

    def get_queryset(self):
        station_id = self.request.query_params.get('station_id', None)

        queryset = StationImage.objects.all()

        if station_id is not None:
            queryset = queryset.filter(station__id=station_id)

        return queryset


class StationFileViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    parser_class = (FileUploadParser,)
    queryset = StationFile.objects.all()
    serializer_class = serializers.StationFileSerializer

    def get_queryset(self):
        station_id = self.request.query_params.get('station_id', None)

        queryset = StationFile.objects.all()

        if station_id is not None:
            queryset = queryset.filter(station__id=station_id)

        return queryset


class DailyMeansView(LoginRequiredMixin, TemplateView):
    template_name = 'wx/products/daily_means.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.values('id', 'name', 'code')

        return self.render_to_response(context)


class RangeThresholdView(LoginRequiredMixin, TemplateView):
    template_name = 'wx/quality_control/range_threshold.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.select_related('profile').all()

        context['station_profile_list'] = StationProfile.objects.all()
        context['station_watershed_list'] = Watershed.objects.all()
        context['station_district_list'] = AdministrativeRegion.objects.all()

        return self.render_to_response(context)


@api_view(['GET', 'POST', 'PATCH', 'DELETE'])
def range_threshold_view(request):
    if request.method == 'GET':

        station_id = request.GET.get('station_id', None)
        variable_id_list = request.GET.get('variable_id_list', None)
        month = request.GET.get('month', None)

        variable_query_statement = ""
        month_query_statement = ""
        query_parameters = {"station_id": station_id, }

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id_list is not None:
            variable_id_list = tuple(json.loads(variable_id_list))
            variable_query_statement = "AND variable_id IN %(variable_id_list)s"
            query_parameters['variable_id_list'] = variable_id_list

        if month is not None:
            month_query_statement = "AND month = %(month)s"
            query_parameters['month'] = month

        get_range_threshold_query = f"""
            SELECT variable.id
                ,variable.name
                ,range_threshold.station_id
                ,range_threshold.range_min
                ,range_threshold.range_max
                ,range_threshold.interval	
                ,range_threshold.month
                ,TO_CHAR(TO_DATE(range_threshold.month::text, 'MM'), 'Month')
                ,range_threshold.id
            FROM wx_qcrangethreshold range_threshold
            JOIN wx_variable variable on range_threshold.variable_id = variable.id 
            WHERE station_id = %(station_id)s
            {variable_query_statement}
            {month_query_statement}
            ORDER BY variable.name, range_threshold.month
        """

        result = []
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_range_threshold_query, query_parameters)

                rows = cursor.fetchall()
                for row in rows:
                    obj = {
                        'variable': {
                            'id': row[0],
                            'name': row[1]
                        },
                        'station_id': row[2],
                        'range_min': row[3],
                        'range_max': row[4],
                        'interval': row[5],
                        'month': row[6],
                        'month_desc': row[7],
                        'id': row[8],

                    }
                    result.append(obj)

        return Response(result, status=status.HTTP_200_OK)


    elif request.method == 'POST':

        station_id = request.data['station_id']
        variable_id = request.data['variable_id']
        month = request.data['month']
        interval = request.data['interval']
        range_min = request.data['range_min']
        range_max = request.data['range_max']

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id is None:
            JsonResponse(data={"message": "'variable_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        if month is None:
            JsonResponse(data={"message": "'month' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if interval is None:
            JsonResponse(data={"message": "'interval' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if range_min is None:
            JsonResponse(data={"message": "'range_min' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if range_max is None:
            JsonResponse(data={"message": "'range_max' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        post_range_threshold_query = f"""
            INSERT INTO wx_qcrangethreshold (created_at, updated_at, range_min, range_max, station_id, variable_id, interval, month) 
            VALUES (now(), now(), %(range_min)s, %(range_max)s , %(station_id)s, %(variable_id)s, %(interval)s, %(month)s)
        """
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(post_range_threshold_query,
                                   {'station_id': station_id, 'variable_id': variable_id, 'month': month,
                                    'interval': interval, 'range_min': range_min, 'range_max': range_max, })
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return JsonResponse(data={"message": "Threshold already exists"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)


    elif request.method == 'PATCH':
        range_threshold_id = request.GET.get('id', None)
        month = request.data['month']
        interval = request.data['interval']
        range_min = request.data['range_min']
        range_max = request.data['range_max']

        if range_threshold_id is None:
            JsonResponse(data={"message": "'id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if month is None:
            JsonResponse(data={"message": "'month' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if interval is None:
            JsonResponse(data={"message": "'interval' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if range_min is None:
            JsonResponse(data={"message": "'range_min' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if range_max is None:
            JsonResponse(data={"message": "'range_max' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        patch_range_threshold_query = f"""
            UPDATE wx_qcrangethreshold
            SET month = %(month)s
               ,interval = %(interval)s
               ,range_min = %(range_min)s
               ,range_max = %(range_max)s
            WHERE id = %(range_threshold_id)s
        """

        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(patch_range_threshold_query,
                                   {'range_threshold_id': range_threshold_id, 'month': month, 'interval': interval,
                                    'range_min': range_min, 'range_max': range_max, })
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return JsonResponse(data={"message": "Threshold already exists"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)


    elif request.method == 'DELETE':
        range_threshold_id = request.GET.get('id', None)

        if range_threshold_id is None:
            JsonResponse(data={"message": "'range_threshold_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        delete_range_threshold_query = f""" DELETE FROM wx_qcrangethreshold WHERE id = %(range_threshold_id)s """
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(delete_range_threshold_query, {'range_threshold_id': range_threshold_id})
                except:
                    conn.rollback()
                    return JsonResponse(data={"message": "Error on delete threshold"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)

    return Response([], status=status.HTTP_200_OK)


class StepThresholdView(LoginRequiredMixin, TemplateView):
    template_name = 'wx/quality_control/step_threshold.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.select_related('profile').all()

        context['station_profile_list'] = StationProfile.objects.all()
        context['station_watershed_list'] = Watershed.objects.all()
        context['station_district_list'] = AdministrativeRegion.objects.all()

        return self.render_to_response(context)


@api_view(['GET', 'POST', 'PATCH', 'DELETE'])
def step_threshold_view(request):
    if request.method == 'GET':

        station_id = request.GET.get('station_id', None)
        variable_id_list = request.GET.get('variable_id_list', None)

        variable_query_statement = ""
        query_parameters = {"station_id": station_id, }

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id_list is not None:
            variable_id_list = tuple(json.loads(variable_id_list))
            variable_query_statement = "AND variable_id IN %(variable_id_list)s"
            query_parameters['variable_id_list'] = variable_id_list

        get_step_threshold_query = f"""
            SELECT variable.id
                ,variable.name
                ,step_threshold.station_id
                ,step_threshold.step_min
                ,step_threshold.step_max
                ,step_threshold.interval	
                ,step_threshold.id
            FROM wx_qcstepthreshold step_threshold
            JOIN wx_variable variable on step_threshold.variable_id = variable.id 
            WHERE station_id = %(station_id)s
            {variable_query_statement}
            ORDER BY variable.name
        """

        result = []
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_step_threshold_query, query_parameters)

                rows = cursor.fetchall()
                for row in rows:
                    obj = {
                        'variable': {
                            'id': row[0],
                            'name': row[1]
                        },
                        'station_id': row[2],
                        'step_min': row[3],
                        'step_max': row[4],
                        'interval': row[5],
                        'id': row[6],

                    }
                    result.append(obj)

        return Response(result, status=status.HTTP_200_OK)


    elif request.method == 'POST':

        station_id = request.data['station_id']
        variable_id = request.data['variable_id']
        interval = request.data['interval']
        step_min = request.data['step_min']
        step_max = request.data['step_max']

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id is None:
            JsonResponse(data={"message": "'variable_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        if interval is None:
            JsonResponse(data={"message": "'interval' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if step_min is None:
            JsonResponse(data={"message": "'step_min' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if step_max is None:
            JsonResponse(data={"message": "'step_max' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        post_step_threshold_query = f"""
            INSERT INTO wx_qcstepthreshold (created_at, updated_at, step_min, step_max, station_id, variable_id, interval) 
            VALUES (now(), now(), %(step_min)s, %(step_max)s , %(station_id)s, %(variable_id)s, %(interval)s)
        """
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(post_step_threshold_query,
                                   {'station_id': station_id, 'variable_id': variable_id, 'interval': interval,
                                    'step_min': step_min, 'step_max': step_max, })
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return JsonResponse(data={"message": "Threshold already exists"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)


    elif request.method == 'PATCH':
        step_threshold_id = request.GET.get('id', None)
        interval = request.data['interval']
        step_min = request.data['step_min']
        step_max = request.data['step_max']

        if step_threshold_id is None:
            JsonResponse(data={"message": "'id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if interval is None:
            JsonResponse(data={"message": "'interval' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if step_min is None:
            JsonResponse(data={"message": "'step_min' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if step_max is None:
            JsonResponse(data={"message": "'step_max' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        patch_step_threshold_query = f"""
            UPDATE wx_qcstepthreshold
            SET interval = %(interval)s
               ,step_min = %(step_min)s
               ,step_max = %(step_max)s
            WHERE id = %(step_threshold_id)s
        """

        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(patch_step_threshold_query,
                                   {'step_threshold_id': step_threshold_id, 'interval': interval, 'step_min': step_min,
                                    'step_max': step_max, })
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return JsonResponse(data={"message": "Threshold already exists"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)


    elif request.method == 'DELETE':
        step_threshold_id = request.GET.get('id', None)

        if step_threshold_id is None:
            JsonResponse(data={"message": "'step_threshold_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        delete_step_threshold_query = f""" DELETE FROM wx_qcstepthreshold WHERE id = %(step_threshold_id)s """
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(delete_step_threshold_query, {'step_threshold_id': step_threshold_id})
                except:
                    conn.rollback()
                    return JsonResponse(data={"message": "Error on delete threshold"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)

    return Response([], status=status.HTTP_200_OK)


class PersistThresholdView(LoginRequiredMixin, TemplateView):
    template_name = 'wx/quality_control/persist_threshold.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['station_list'] = Station.objects.select_related('profile').all()

        context['station_profile_list'] = StationProfile.objects.all()
        context['station_watershed_list'] = Watershed.objects.all()
        context['station_district_list'] = AdministrativeRegion.objects.all()

        return self.render_to_response(context)


@api_view(['GET', 'POST', 'PATCH', 'DELETE'])
def persist_threshold_view(request):
    if request.method == 'GET':

        station_id = request.GET.get('station_id', None)
        variable_id_list = request.GET.get('variable_id_list', None)

        variable_query_statement = ""
        query_parameters = {"station_id": station_id, }

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id_list is not None:
            variable_id_list = tuple(json.loads(variable_id_list))
            variable_query_statement = "AND variable_id IN %(variable_id_list)s"
            query_parameters['variable_id_list'] = variable_id_list

        get_persist_threshold_query = f"""
            SELECT variable.id
                ,variable.name
                ,persist_threshold.station_id
                ,persist_threshold.window
                ,persist_threshold.minimum_variance
                ,persist_threshold.interval	
                ,persist_threshold.id
            FROM wx_qcpersistthreshold persist_threshold
            JOIN wx_variable variable on persist_threshold.variable_id = variable.id 
            WHERE station_id = %(station_id)s
            {variable_query_statement}
            ORDER BY variable.name
        """

        result = []
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                cursor.execute(get_persist_threshold_query, query_parameters)

                rows = cursor.fetchall()
                for row in rows:
                    obj = {
                        'variable': {
                            'id': row[0],
                            'name': row[1]
                        },
                        'station_id': row[2],
                        'window': row[3],
                        'minimum_variance': row[4],
                        'interval': row[5],
                        'id': row[6],

                    }
                    result.append(obj)

        return Response(result, status=status.HTTP_200_OK)


    elif request.method == 'POST':

        station_id = request.data['station_id']
        variable_id = request.data['variable_id']
        interval = request.data['interval']
        window = request.data['window']
        minimum_variance = request.data['minimum_variance']

        if station_id is None:
            JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if variable_id is None:
            JsonResponse(data={"message": "'variable_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        if interval is None:
            JsonResponse(data={"message": "'interval' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if window is None:
            JsonResponse(data={"message": "'window' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if minimum_variance is None:
            JsonResponse(data={"message": "'minimum_variance' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        post_persist_threshold_query = f"""
            INSERT INTO wx_qcpersistthreshold (created_at, updated_at, "window", minimum_variance, station_id, variable_id, interval) 
            VALUES (now(), now(), %(window)s, %(minimum_variance)s , %(station_id)s, %(variable_id)s, %(interval)s)
        """
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(post_persist_threshold_query,
                                   {'station_id': station_id, 'variable_id': variable_id, 'interval': interval,
                                    'window': window, 'minimum_variance': minimum_variance, })
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return JsonResponse(data={"message": "Threshold already exists"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)


    elif request.method == 'PATCH':
        persist_threshold_id = request.GET.get('id', None)
        interval = request.data['interval']
        window = request.data['window']
        minimum_variance = request.data['minimum_variance']

        if persist_threshold_id is None:
            JsonResponse(data={"message": "'id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if interval is None:
            JsonResponse(data={"message": "'interval' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if window is None:
            JsonResponse(data={"message": "'window' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

        if minimum_variance is None:
            JsonResponse(data={"message": "'minimum_variance' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        patch_persist_threshold_query = f"""
            UPDATE wx_qcpersistthreshold
            SET interval = %(interval)s
               ,"window" = %(window)s
               ,minimum_variance = %(minimum_variance)s
            WHERE id = %(persist_threshold_id)s
        """

        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(patch_persist_threshold_query,
                                   {'persist_threshold_id': persist_threshold_id, 'interval': interval,
                                    'window': window, 'minimum_variance': minimum_variance, })
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return JsonResponse(data={"message": "Threshold already exists"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)


    elif request.method == 'DELETE':
        persist_threshold_id = request.GET.get('id', None)

        if persist_threshold_id is None:
            JsonResponse(data={"message": "'persist_threshold_id' parameter cannot be null."},
                         status=status.HTTP_400_BAD_REQUEST)

        delete_persist_threshold_query = f""" DELETE FROM wx_qcpersistthreshold WHERE id = %(persist_threshold_id)s """
        with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(delete_persist_threshold_query, {'persist_threshold_id': persist_threshold_id})
                except:
                    conn.rollback()
                    return JsonResponse(data={"message": "Error on delete threshold"},
                                        status=status.HTTP_400_BAD_REQUEST)

            conn.commit()
        return Response(status=status.HTTP_200_OK)

    return Response([], status=status.HTTP_200_OK)


@api_view(['GET'])
def daily_means_data_view(request):
    station_id = request.GET.get('station_id', None)
    month = request.GET.get('month', None)
    variable_id_list = request.GET.get('variable_id_list', None)
    begin_year = request.GET.get('begin_year', None)
    end_year = request.GET.get('end_year', None)
    filter_year_query = ""
    filter_year_query_avg = ""
    period = "All years"

    if station_id is None:
        JsonResponse(data={"message": "'station_id' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

    if month is None:
        JsonResponse(data={"message": "'month' parameter cannot be null."}, status=status.HTTP_400_BAD_REQUEST)

    if variable_id_list is not None:
        variable_id_list = json.loads(variable_id_list)
    else:
        JsonResponse(data={"message": "'variable_id_list' parameter cannot be null."},
                     status=status.HTTP_400_BAD_REQUEST)

    if begin_year is not None and end_year is not None:
        try:
            filter_year_query = "AND EXTRACT(YEAR from day) >= %(begin_year)s AND EXTRACT(YEAR from day) <= %(end_year)s"
            period = f"{begin_year} - {end_year}"

            begin_year = int(begin_year)
            end_year = int(end_year)
        except ValueError:
            JsonResponse(data={"message": "Invalid 'begin_year' or 'end_year' parameters."},
                         status=status.HTTP_400_BAD_REQUEST)

    month = int(month)
    res = {}
    variable_dict = {}
    variable_symbol_dict = {}
    for variable_id in variable_id_list:
        query_params_dict = {"station_id": station_id, "month": month, "variable_id": variable_id,
                             "begin_year": begin_year, "end_year": end_year}
        variable = Variable.objects.get(pk=variable_id)

        variable_symbol_dict[variable.symbol] = variable.name
        aggregation_type = variable.sampling_operation.name if variable.sampling_operation is not None else None
        data_dict = {}
        summary_dict = {}
        colspan = 1
        headers = ['Average']
        columns = ['value']

        if aggregation_type == 'Accumulation':
            colspan = 3
            headers = ['Greatest', 'Year', 'Years']
            columns = ['agg_value', 'year', 'years']

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT data.day
                          ,data.sum_value
                          ,data.year
                          ,data.years
                    FROM (SELECT EXTRACT(DAY from day) as day
                                ,sum_value
                                ,EXTRACT(YEAR from day) as year
                                ,RANK () OVER (PARTITION BY EXTRACT(DAY from day) ORDER BY sum_value DESC) as rank
                                ,count(1) OVER (PARTITION BY EXTRACT(DAY from day)) as years
                         FROM daily_summary
                         WHERE station_id  = %(station_id)s
                           AND EXTRACT(MONTH from day) = %(month)s
                           AND variable_id = %(variable_id)s
                           {filter_year_query}) data
                    WHERE data.rank = 1
                """, query_params_dict)
                rows = cursor.fetchall()

                if len(rows) > 0:
                    df = pd.DataFrame(data=rows, columns=("day", "sum_value", "year", "years"))
                    for index, row in df.iterrows():
                        data_dict[int(row['day'])] = {"agg_value": round(row['sum_value'], 2), "year": row['year'],
                                                      "years": row['years']}

                    max_row = df.loc[df["sum_value"].idxmax()]
                    summary_dict = {"agg_value": round(max_row['sum_value'], 2), "year": max_row['year']}


        elif aggregation_type == 'Maximum':
            colspan = 4
            headers = ['Average', 'Extreme', 'Year', 'Years']
            columns = ['value', 'agg_value', 'year', 'years']

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT data.day
                          ,summary.value
                          ,data.max_value
                          ,data.year
                          ,data.years
                    FROM (SELECT EXTRACT(DAY from day) as day
                                ,max_value
                                ,EXTRACT(YEAR from day) as year
                                ,RANK () OVER (PARTITION BY EXTRACT(DAY from day) ORDER BY max_value DESC) as rank
                                ,count(1) OVER (PARTITION BY EXTRACT(DAY from day)) as years
                         FROM daily_summary
                         WHERE station_id  = %(station_id)s
                           AND EXTRACT(MONTH from day) = %(month)s
                           AND variable_id = %(variable_id)s
                           {filter_year_query}) data,
                        (SELECT EXTRACT(DAY from day) as day
                                ,avg(max_value) as value
                         FROM daily_summary
                         WHERE station_id  = %(station_id)s
                           AND EXTRACT(MONTH from day) = %(month)s
                           AND variable_id = %(variable_id)s
                           {filter_year_query}
                         GROUP BY 1) summary
                    WHERE data.rank = 1
                    AND summary.day = data.day
                """, query_params_dict)
                rows = cursor.fetchall()

                if len(rows) > 0:
                    df = pd.DataFrame(data=rows, columns=("day", "value", "max_value", "year", "years"))
                    for index, row in df.iterrows():
                        data_dict[int(row['day'])] = {"value": round(row['value'], 2),
                                                      "agg_value": round(row['max_value'], 2), "year": row['year'],
                                                      "years": row['years']}

                    max_row = df.loc[df["max_value"].idxmax()]
                    summary_dict = {"value": round(df["value"].mean(), 2), "agg_value": round(max_row['max_value'], 2),
                                    "year": max_row['year']}

        elif aggregation_type == 'Minimum':
            headers = ['Average', 'Extreme', 'Year', 'Years']
            columns = ['value', 'agg_value', 'year', 'years']
            colspan = 4
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT data.day
                          ,summary.value
                          ,data.min_value
                          ,data.year
                          ,data.years
                    FROM (SELECT EXTRACT(DAY from day) as day
                                ,min_value
                                ,EXTRACT(YEAR from day) as year
                                ,RANK () OVER (PARTITION BY EXTRACT(DAY from day) ORDER BY min_value ASC) as rank
                                ,count(1) OVER (PARTITION BY EXTRACT(DAY from day)) as years
                         FROM daily_summary
                         WHERE station_id  = %(station_id)s
                           AND EXTRACT(MONTH from day) = %(month)s
                           AND variable_id = %(variable_id)s
                           {filter_year_query}) data,
                        (SELECT EXTRACT(DAY from day) as day
                                ,avg(min_value) as value
                         FROM daily_summary
                         WHERE station_id  = %(station_id)s
                           AND EXTRACT(MONTH from day) = %(month)s
                           AND variable_id = %(variable_id)s
                           {filter_year_query}
                         GROUP BY 1) summary
                    WHERE data.rank = 1
                    AND summary.day = data.day
                """, query_params_dict)
                rows = cursor.fetchall()

                if len(rows) > 0:
                    df = pd.DataFrame(data=rows, columns=("day", "value", "min_value", "year", "years"))
                    for index, row in df.iterrows():
                        data_dict[int(row['day'])] = {"value": round(row['value'], 2),
                                                      "agg_value": round(row['min_value'], 2), "year": row['year'],
                                                      "years": row['years']}

                    max_row = df.loc[df["min_value"].idxmin()]
                    summary_dict = {"value": round(df["value"].mean(), 2), "agg_value": round(max_row['min_value'], 2),
                                    "year": max_row['year']}

        else:
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT EXTRACT(DAY from day)
                          ,avg(avg_value)
                    FROM daily_summary data
                    WHERE station_id = %(station_id)s
                    AND EXTRACT(MONTH from day) = %(month)s
                    AND variable_id = %(variable_id)s
                    {filter_year_query}
                    GROUP BY 1
                """, query_params_dict)
                rows = cursor.fetchall()

                if len(rows) > 0:
                    for row in rows:
                        day = int(row[0])
                        value = round(row[1], 2)
                        data_dict[day] = {"value": value}

                    df = pd.DataFrame(data=rows, columns=("day", "value"))
                    summary_dict = {"value": round(df["value"].mean(), 2)}

        variable_dict[variable_id] = {
            'metadata': {
                "name": variable.symbol,
                "id": variable.id,
                "unit": variable.unit.name if variable.unit is not None else "",
                "unit_symbol": variable.unit.symbol if variable.unit is not None else "",
                "aggregation": aggregation_type,
                "colspan": colspan,
                "headers": headers,
                "columns": columns
            },
            'data': data_dict,
            'summary': summary_dict,
        }

    res['variables'] = variable_dict
    station = Station.objects.get(pk=station_id)
    res['station'] = {
        "name": station.name,
        "district": station.region,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation": station.elevation,
        "variables": variable_symbol_dict,
    }

    res['params'] = {
        "month": month,
        "period": period,
    }

    return JsonResponse(res, status=status.HTTP_200_OK)


class DataInventoryView(LoginRequiredMixin, TemplateView):
    template_name = "wx/data_inventory.html"

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)

        context['variable_list'] = Variable.objects.all()

        return self.render_to_response(context)


@api_view(['GET'])
def get_data_inventory(request):
    start_date = request.GET.get('start_date', None)
    end_date = request.GET.get('end_date', None)

    try:
        start_date = datetime.datetime.strptime(start_date, '%Y')
        end_date = datetime.datetime.strptime(end_date, '%Y')

        if start_date >= end_date:
            return JsonResponse({"message": "Invalid date. End date must be greater than start date"},
                                status=status.HTTP_400_BAD_REQUEST)

    except ValueError:
        return JsonResponse({"message": "Invalid date format. The expected date format is 'YYYY'"},
                            status=status.HTTP_400_BAD_REQUEST)

    result = []

    query = """
        SELECT DATE_TRUNC('YEAR', station_data.datetime)
              ,station.id
              ,station.name
              ,station.code
              ,TRUNC(AVG(station_data.record_count_percentage)::numeric, 2)
        FROM wx_stationdataminimuminterval AS station_data
        JOIN wx_station AS station ON station.id = station_data.station_id
        WHERE station_data.datetime >= %(start_date)s
          AND station_data.datetime <  %(end_date)s
        GROUP BY 1, station.id, station.name
        ORDER BY station.name
    """

    with connection.cursor() as cursor:
        cursor.execute(query, {"start_date": start_date, "end_date": end_date})
        rows = cursor.fetchall()

        for row in rows:
            obj = {
                'datetime': row[0],
                'station': {
                    'id': row[1],
                    'name': row[2],
                    'code': row[3],
                },
                'percentage': row[4],
            }
            result.append(obj)

    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_station_data_inventory(request):
    start_date = request.GET.get('start_date', None)
    end_date = request.GET.get('end_date', None)
    station_id = request.GET.get('station_id', None)

    try:
        start_date = datetime.datetime.strptime(start_date, '%Y')
        end_date = datetime.datetime.strptime(end_date, '%Y')

        if start_date >= end_date:
            return JsonResponse({"message": "Invalid date. End date must be greater than start date"},
                                status=status.HTTP_400_BAD_REQUEST)

    except ValueError:
        return JsonResponse({"message": "Invalid date format. The expected date format is 'YYYY'"},
                            status=status.HTTP_400_BAD_REQUEST)

    if station_id is None:
        return JsonResponse({"message": "Invalid request. Station id must be provided"},
                            status=status.HTTP_400_BAD_REQUEST)

    result = []
    query = """
        SELECT DATE_TRUNC('YEAR', station_data.datetime)
              ,station.id
              ,station.name
              ,station.code
              ,variable.id
              ,variable.name
              ,TRUNC(AVG(station_data.record_count_percentage)::numeric, 2)
        FROM wx_stationdataminimuminterval AS station_data
        JOIN wx_station AS station ON station.id = station_data.station_id
        JOIN wx_variable AS variable ON variable.id = station_data.variable_id
        WHERE station_data.station_id = %(station_id)s
          AND station_data.datetime  >= %(start_date)s
          AND station_data.datetime  <  %(end_date)s
        GROUP BY 1, station.id, station.name, station.code, variable.id, variable.name
        ORDER BY variable.name
    """

    with connection.cursor() as cursor:
        cursor.execute(query, {"station_id": station_id, "start_date": start_date, "end_date": end_date})
        rows = cursor.fetchall()

        for row in rows:
            obj = {
                'datetime': row[0],
                'station': {
                    'id': row[1],
                    'name': row[2],
                    'code': row[3],
                },
                'variable': {
                    'id': row[4],
                    'name': row[5],
                },
                'percentage': row[6],
            }
            result.append(obj)

    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_station_variable_data_inventory(request):
    start_date = request.GET.get('start_date', None)
    end_date = request.GET.get('end_date', None)
    station_id = request.GET.get('station_id', None)
    variable_id = request.GET.get('variable_id', None)

    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        if start_date >= end_date:
            return JsonResponse({"message": "Invalid date. End date must be greater than start date"},
                                status=status.HTTP_400_BAD_REQUEST)

    except ValueError:
        return JsonResponse({"message": "Invalid date format. The expected date format is 'YYYY-MM-DDTHH:MI:SS.sssZ'"},
                            status=status.HTTP_400_BAD_REQUEST)

    if station_id is None:
        return JsonResponse({"message": "Invalid request. Station id must be provided"},
                            status=status.HTTP_400_BAD_REQUEST)

    if variable_id is None:
        return JsonResponse({"message": "Invalid request. Variable id must be provided"},
                            status=status.HTTP_400_BAD_REQUEST)

    result = []
    query = """
        SELECT EXTRACT('MONTH' FROM station_data.datetime) AS month
              ,EXTRACT('DAY' FROM station_data.datetime) AS day
              ,station_data.datetime
              ,measurement_variable.name
              ,TRUNC(AVG(station_data.record_count_percentage)::numeric, 2)
        FROM wx_stationdataminimuminterval AS station_data
        JOIN wx_variable variable ON station_data.variable_id=variable.id
        JOIN wx_measurementvariable measurement_variable ON variable.measurement_variable_id=measurement_variable.id
        WHERE station_data.variable_id = %(variable_id)s
          AND station_data.station_id  = %(station_id)s
          AND station_data.datetime   >= %(start_date)s
          AND station_data.datetime   <= %(end_date)s
        GROUP BY 1, 2, station_data.datetime, measurement_variable.name

    """

    with connection.cursor() as cursor:
        cursor.execute(query, {"start_date": start_date, "end_date": end_date, "variable_id": variable_id,
                               "station_id": station_id})
        rows = cursor.fetchall()

        for row in rows:
            obj = {
                'month': row[0],
                'day': row[1],
                'datetime': row[2],
                'measurement_variable': slugify(row[3]),
                'percentage': row[4],
            }
            result.append(obj)

    return Response(result, status=status.HTTP_200_OK)


@api_view(['POST'])
def delete_pgia_hourly_capture_row(request):
    request_date = request.data['date']
    hour = request.data['hour']
    station_id = request.data['station_id']
    variable_id_list = request.data['variable_ids']

    try:
        request_date = datetime.datetime.strptime(request_date, '%Y-%m-%d')
    except ValueError:
        return JsonResponse({"message": "Invalid date format. The expected date format is 'YYYY-MM-DD'"},
                            status=status.HTTP_400_BAD_REQUEST)

    if station_id is None:
        return JsonResponse({"message": "Invalid request. Station id must be provided"},
                            status=status.HTTP_400_BAD_REQUEST)

    if hour is None:
        return JsonResponse({"message": "Invalid request. Hour must be provided"}, status=status.HTTP_400_BAD_REQUEST)

    if variable_id_list is None:
        return JsonResponse({"message": "Invalid request. Variable ids must be provided"},
                            status=status.HTTP_400_BAD_REQUEST)

    variable_id_list = tuple(variable_id_list)
    station = Station.objects.get(id=station_id)
    datetime_offset = pytz.FixedOffset(station.utc_offset_minutes)
    current_datetime = datetime_offset.localize(request_date.replace(hour=hour))

    result = []
    delete_query = """
        DELETE FROM raw_data
        WHERE station_id = %(station_id)s
          AND variable_id IN %(variable_id_list)s
          AND datetime = %(current_datetime)s
    """

    get_last_updated_datetime_query = """
        SELECT max(last_data_datetime)
        FROM wx_stationvariable
        WHERE station_id = %(station_id)s
          AND variable_id IN %(variable_id_list)s
        ORDER BY 1 DESC
    """

    update_last_updated_datetime_query = """
        WITH rd as (
            SELECT station_id
                  ,variable_id
                  ,measured
                  ,code
                  ,datetime
                  ,RANK() OVER (PARTITION BY station_id, variable_id ORDER BY datetime DESC) datetime_rank
            FROM raw_data
            WHERE station_id = %(station_id)s
              AND variable_id IN %(variable_id_list)s)
        UPDATE wx_stationvariable sv
        SET last_data_datetime = rd.datetime
           ,last_data_value    = rd.measured
           ,last_data_code     = rd.code
        FROM rd
        WHERE sv.station_id = rd.station_id
          AND sv.variable_id = rd.variable_id
          AND rd.datetime_rank = 1
    """

    create_daily_summary_task_query = """
        INSERT INTO wx_dailysummarytask (station_id, date, created_at, updated_at)
        VALUES (%(station_id)s, %(current_datetime)s, now(), now())
        ON CONFLICT DO NOTHING
    """

    create_hourly_summary_task_query = """
        INSERT INTO wx_hourlysummarytask (station_id, datetime, created_at, updated_at)
        VALUES (%(station_id)s, %(current_datetime)s, now(), now())
        ON CONFLICT DO NOTHING
    """

    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:
            cursor.execute(delete_query, {"station_id": station_id, "variable_id_list": variable_id_list,
                                          "current_datetime": current_datetime})

            cursor.execute(create_daily_summary_task_query,
                           {"station_id": station_id, "current_datetime": request_date})
            cursor.execute(create_hourly_summary_task_query,
                           {"station_id": station_id, "current_datetime": current_datetime})

            cursor.execute(get_last_updated_datetime_query,
                           {"station_id": station_id, "variable_id_list": variable_id_list})

            last_data_datetime_row = cursor.fetchone()
            if last_data_datetime_row is not None:
                last_data_datetime = last_data_datetime_row[0]

                if last_data_datetime == current_datetime:
                    cursor.execute(update_last_updated_datetime_query,
                                   {"station_id": station_id, "variable_id_list": variable_id_list})
        conn.commit()

    return Response(result, status=status.HTTP_200_OK)

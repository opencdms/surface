import logging

import psycopg2
from django.conf import settings

logger = logging.getLogger('surface')


def get_surface_context(req):
    return {
        'TIMEZONE_NAME': settings.TIMEZONE_NAME,
        'MISSING_VALUE': settings.MISSING_VALUE,
        'MAP_LATITUDE': settings.MAP_LATITUDE,
        'MAP_LONGITUDE': settings.MAP_LONGITUDE,
        'MAP_ZOOM': settings.MAP_ZOOM,
        'SPATIAL_ANALYSIS_INITIAL_LATITUDE': settings.SPATIAL_ANALYSIS_INITIAL_LATITUDE,
        'SPATIAL_ANALYSIS_INITIAL_LONGITUDE': settings.SPATIAL_ANALYSIS_INITIAL_LONGITUDE,
        'SPATIAL_ANALYSIS_FINAL_LATITUDE': settings.SPATIAL_ANALYSIS_FINAL_LATITUDE,
        'SPATIAL_ANALYSIS_FINAL_LONGITUDE': settings.SPATIAL_ANALYSIS_FINAL_LONGITUDE,
        'STATION_MAP_WIND_SPEED_ID': settings.STATION_MAP_WIND_SPEED_ID,
        'STATION_MAP_WIND_GUST_ID': settings.STATION_MAP_WIND_GUST_ID,
        'STATION_MAP_WIND_DIRECTION_ID': settings.STATION_MAP_WIND_DIRECTION_ID,
        'STATION_MAP_TEMP_MAX_ID': settings.STATION_MAP_TEMP_MAX_ID,
        'STATION_MAP_TEMP_MIN_ID': settings.STATION_MAP_TEMP_MIN_ID,
        'STATION_MAP_TEMP_AVG_ID': settings.STATION_MAP_TEMP_AVG_ID,
        'STATION_MAP_ATM_PRESSURE_ID': settings.STATION_MAP_ATM_PRESSURE_ID,
        'STATION_MAP_PRECIPITATION_ID': settings.STATION_MAP_PRECIPITATION_ID,
        'STATION_MAP_RELATIVE_HUMIDITY_ID': settings.STATION_MAP_RELATIVE_HUMIDITY_ID,
        'STATION_MAP_SOLAR_RADIATION_ID': settings.STATION_MAP_SOLAR_RADIATION_ID,
        'STATION_MAP_FILTER_WATERSHED': settings.STATION_MAP_FILTER_WATERSHED,
        'STATION_MAP_FILTER_REGION': settings.STATION_MAP_FILTER_REGION,
        'STATION_MAP_FILTER_PROFILE': settings.STATION_MAP_FILTER_PROFILE,
        'STATION_MAP_FILTER_COMMUNICATION': settings.STATION_MAP_FILTER_COMMUNICATION,
    }


def get_user_wx_permissions(req):
    with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT DISTINCT url_name, permission
                    FROM wx_wxpermission as perm
                    JOIN wx_wxgrouppermission_permissions as gpp ON gpp.wxpermission_id = perm.id
                    JOIN wx_wxgrouppermission as gp ON gp.id = gpp.wxgrouppermission_id
                    JOIN auth_user_groups as aug ON aug.group_id = gp.group_id 
                    WHERE aug.user_id = %s
                """, (req.user.id,))

            user_permissions = {}
            for row in cursor.fetchall():
                if user_permissions.get(row[0]) is None:
                    user_permissions[row[0]] = []

                user_permissions[row[0]].append(row[1])

    return {'USER_PERMISSIONS': user_permissions, 'USER_IS_ADMIN': 1 if req.user.is_superuser else 0}

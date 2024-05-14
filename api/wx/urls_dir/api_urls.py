from django.urls import path, include
from wx import views
from rest_framework import routers


router = routers.DefaultRouter()
router.register(r'station_images', views.StationImageViewSet)
router.register(r'station_files', views.StationFileViewSet)
router.register(r'quality_flags', views.QualityFlagList)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/stations/metadata/', include(router.urls)),

    path('api/administrative_regions/', views.AdministrativeRegionViewSet.as_view({'get': 'list'})),
    path('api/stations/', views.StationViewSet.as_view({'get': 'list', 'post': 'create', 'put': 'update'})),
    path('api/stations_simple/', views.StationSimpleViewSet.as_view({'get': 'list'})),
    path('api/station_profiles/', views.StationProfileViewSet.as_view({'get': 'list'})),
    path('api/stations_variables/', views.StationVariableViewSet.as_view({'get': 'list'})),
    path('api/stations_variables/stations/', views.StationVariableStationViewSet.as_view({'get': 'list'})),

    path('api/stations/metadata/', views.StationMetadataViewSet.as_view({'get': 'list'})),

    path('api/variables/', views.VariableViewSet.as_view({'get': 'list'})),
    path('api/watersheds/', views.WatershedList.as_view()),
    path('api/station_communications/', views.StationCommunicationList.as_view()),
    path('api/livedata/<str:code>', views.livedata),
    path('api/rawdata/', views.raw_data_list),
    path('api/hourlysummaries/', views.hourly_summary_list),
    path('api/dailysummaries/', views.daily_summary_list),
    path('api/monthlysummaries/', views.monthly_summary_list),
    path('api/yearlysummaries/', views.yearly_summary_list),
    path('api/last24hrsummaries/', views.last24_summary_list),
    path('api/station_telemetry_data/<str:date>', views.station_telemetry_data),
    path('api/station_report/', views.station_report_data, name='station_report_data'),
    path('api/quality_control/', views.qc_list),
    path('api/variable-report/', views.variable_report_data, name='variable-report-data'),
    path('api/raw_data_last_24h/<station_id>/', views.raw_data_last_24h),
    path('api/latest_data/<variable_id>/', views.latest_data),
    path('api/daily_means/', views.daily_means_data_view),
    path('api/data_inventory/', views.get_data_inventory),
    path('api/data_inventory_by_station/', views.get_data_inventory_by_station),
    path('api/station_variable_data_month_inventory/', views.get_station_variable_month_data_inventory),
    path('api/station_variable_data_day_inventory/', views.get_station_variable_day_data_inventory),
    path('api/range_threshold/', views.range_threshold_view),

    path('', include('wx.urls_dir.spectacular')),
]

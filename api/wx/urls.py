from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from rest_framework import routers

from django.contrib.auth.decorators import login_required

from wx import views

router = routers.DefaultRouter()
router.register(r'station_images', views.StationImageViewSet)
router.register(r'station_files', views.StationFileViewSet)
router.register(r'quality_flags', views.QualityFlagList)
router.register(r'stations_metadata', views.StationMetadataViewSet)

urlpatterns = [
    path('api/stations/metadata', include(router.urls)),
    path('api/administrative_regions/', views.AdministrativeRegionViewSet.as_view({'get': 'list'})),
    path('api/stations/', views.StationViewSet.as_view({'get': 'list', 'post': 'create', 'put': 'update'})),
    path('api/stations_simple/', views.StationSimpleViewSet.as_view({'get': 'list'})),
    path('api/station_profiles/', views.StationProfileViewSet.as_view({'get': 'list'})),
    path('api/stations_variables/', views.StationVariableViewSet.as_view({'get': 'list'})),
    path('api/stations_variables/stations/', views.StationVariableStationViewSet.as_view({'get': 'list'})),
    path('api/variables/', views.VariableViewSet.as_view({'get': 'list'})),
    path('api/watersheds/', views.WatershedList.as_view()),
    path('api/station_communications/', views.StationCommunicationList.as_view()),
    path('api/livedata/<code>/', views.livedata),
    path('api/rawdata/', views.raw_data_list),
    path('api/hourlysummaries/', views.hourly_summary_list),
    path('api/dailysummaries/', views.daily_summary_list),
    path('api/monthlysummaries/', views.monthly_summary_list),
    path('api/yearlysummaries/', views.yearly_summary_list),
    path('api/last24hrsummaries/', views.last24_summary_list),
    path('api/station_telemetry_data/<str:date>', views.station_telemetry_data),
    path('api/', include(router.urls)),
    path('', views.StationsMapView.as_view(), name='stations-map'),
    path('station_geo_features/<str:lon>/<str:lat>', views.station_geo_features),
    path('decoders/', views.DecoderList.as_view()),
    path('interpolation/', views.interpolate_endpoint),
    path('capture_forms_values/', views.capture_forms_values_get),
    path('capture_forms_values_patch/', views.capture_forms_values_patch),
    path('wx/stations/', views.StationListView.as_view(), name='stations-list'),
    path('wx/stations/map/', views.StationsMapView.as_view(), name='stations-map'),
    path('wx/stations/<int:pk>/', views.StationDetailView.as_view(), name='station-detail'),
    path('wx/stations/metadata/', views.StationMetadataView.as_view(), name='station-metadata'),
    path('wx/stations/create/', views.StationCreate.as_view(), name='station-create'),
    path('wx/stations/<int:pk>/update/', views.StationUpdate.as_view(), name='station-update'),
    path('wx/stations/<int:pk>/delete/', views.StationDelete.as_view(), name='station-delete'),
    path('wx/stations/<int:pk>/files/', views.StationFileList.as_view(), name='stationfiles-list'),
    path('wx/stations/<int:pk>/files/create/', views.StationFileCreate.as_view(), name='stationfile-create'),
    path('wx/stations/<int:pk_station>/files/<int:pk>/delete/', views.StationFileDelete.as_view(),
         name='stationfile-delete'),
    path('wx/stations/<int:pk>/variables/', views.StationVariableListView.as_view(), name='stationvariable-list'),
    path('wx/stations/<int:pk>/variables/create/', views.StationVariableCreateView.as_view(),
         name='stationvariable-create'),
    path('wx/stations/<int:pk_station>/variables/<int:pk>/delete/', views.StationVariableDeleteView.as_view(),
         name='stationvariable-delete'),
    path('wx/products/station_report/', views.StationReportView.as_view(), name='station-report'),
    path('api/station_report/', views.station_report_data, name='station_report_data'),
    path('wx/variablereport/', views.VariableReportView.as_view(), name='variable-report'),
    path('wx/product/compare/', views.ProductCompareView.as_view(), name='product-compare'),
    path('wx/quality_control/validation/', views.QualityControlView.as_view(), name='quality-control'),
    path('api/quality_control/', views.qc_list),
    path('api/variable-report/', views.variable_report_data, name='variable-report-data'),
    path('wx/data/capture/', views.DataCaptureView, name='data-capture'),
    path('wx/data/export/', views.DataExportView.as_view(), name='data-export'),
    path('wx/data/export/files/', views.DataExportFiles, name='data-export-files'),
    path('wx/data/export/download/', views.DownloadDataFile, name='data-export-download'),
    path('wx/data/export/delete/', views.DeleteDataFile, name='data-export-delete'),
    path('wx/data/export/schedule/', views.ScheduleDataExport, name='data-export-schedule'),
    path('get_yearly_average/', views.get_yearly_average),
    path('wx/reports/yearly_average/', views.YearlyAverageReport.as_view(), name='yearly-average'),
    path('wx/reports/synop_capture/', views.SynopCaptureView.as_view(), name='synop-capture'),
    path('wx/reports/synop_capture/load/', views.pgia_load, name='load-pgia-report'),
    path('wx/reports/synop_capture/update/', views.pgia_update, name='update-pgia-report'),
    path('wx/reports/synop_capture/delete/', views.delete_pgia_hourly_capture_row, name='delete-pgia-report-row'),
    path('wx/data/capture/daily/', views.DailyFormView, name='daily-form'),
    path('wx/data/capture/daily/load/', views.MonthlyFormLoad, name='load-monthly-form'),
    path('wx/data/capture/daily/update/', views.MonthlyFormUpdate, name='update-monthly-form'),
    path('wx/spatial_analysis/', views.SpatialAnalysisView.as_view(), name='spatial-analysis'),
    path('wx/spatial_analysis/image', views.GetInterpolationImage, name='spatial-analysis-image'),
    path('wx/spatial_analysis/data', views.GetInterpolationData, name='spatial-analysis-data'),
    path('wx/spatial_analysis/interpolate_data', views.InterpolatePostData, name='spatial-analysis-interpolate_data'),
    path('wx/spatial_analysis/get_image', views.GetImage, name='spatial-analysis-get-image'),
    path('wx/spatial_analysis/color_bar', views.GetColorMapBar, name='spatial-analysis-color-bar'),
    path('coming-soon', views.ComingSoonView.as_view(), name='coming-soon'),
    path('coming-soon-qc', views.ComingSoonView.as_view(), name='coming-soon-qc'),
    path('api/raw_data_last_24h/<station_id>/', views.raw_data_last_24h),
    path('api/latest_data/<variable_id>/', views.latest_data),
    path('wx/product/extremes_means/', views.ExtremesMeansView.as_view(), name='extremes-means'),
    path('api/daily_means/', views.daily_means_data_view),
    path('wx/data/inventory/', views.DataInventoryView.as_view(), name='data-inventory'),
    path('api/data_inventory/', views.get_data_inventory),
    path('api/data_inventory_by_station/', views.get_data_inventory_by_station),
    path('api/station_variable_data_month_inventory/', views.get_station_variable_month_data_inventory),
    path('api/station_variable_data_day_inventory/', views.get_station_variable_day_data_inventory),
    path('api/range_threshold/', views.range_threshold_view), # For synop and daily data capture
    path('wx/quality_control/update_reference_station/', views.update_reference_station),    
    path('wx/quality_control/global_threshold/update/', views.update_global_threshold),
    path('wx/quality_control/range_threshold/', views.get_range_threshold_form, name='range-threshold'),
    path('wx/quality_control/range_threshold/get/', views.get_range_threshold),
    path('wx/quality_control/range_threshold/update/', views.update_range_threshold),    
    path('wx/quality_control/range_threshold/delete/', views.delete_range_threshold),    
    path('wx/quality_control/step_threshold/', views.get_step_threshold_form, name='step-threshold'),
    path('wx/quality_control/step_threshold/get/', views.get_step_threshold),
    path('wx/quality_control/step_threshold/update/', views.update_step_threshold),
    path('wx/quality_control/step_threshold/delete/', views.delete_step_threshold),
    path('wx/quality_control/persist_threshold/', views.get_persist_threshold_form, name='persist-threshold'),
    path('wx/quality_control/persist_threshold/get/', views.get_persist_threshold),
    path('wx/quality_control/persist_threshold/update/', views.update_persist_threshold),
    path('wx/quality_control/persist_threshold/delete/', views.delete_persist_threshold),
    path('wx/maintenance_report/', login_required(views.get_maintenance_reports), name='maintenance-reports'),
    path('wx/maintenance_report/get_reports/', login_required(views.get_maintenance_report_list)),
    path('wx/maintenance_report/new_report/', login_required(views.get_maintenance_report_form), name='new-maintenance-report'),
    path('wx/maintenance_report/create/', login_required(views.create_maintenance_report)),
    path('wx/maintenance_report/<int:id>/get/', login_required(views.get_maintenance_report)),
    path('wx/maintenance_report/<int:id>/update/', login_required(views.update_maintenance_report)),
    path('wx/maintenance_report/<int:id>/update/condition/', login_required(views.update_maintenance_report_condition)),
    path('wx/maintenance_report/<int:id>/update/contacts/', login_required(views.update_maintenance_report_contacts)),
    path('wx/maintenance_report/<int:id>/update/summary/', login_required(views.update_maintenance_report_summary)),
    path('wx/maintenance_report/<int:id>/update/datalogger/', login_required(views.update_maintenance_report_datalogger)),    
    path('wx/maintenance_report/<int:id>/delete/', login_required(views.delete_maintenance_report)),
    path('wx/maintenance_report/<int:id>/approve/', login_required(views.approve_maintenance_report)),
    # path('wx/maintenance_report/<int:id>/view/<int:source>/', login_required(views.get_maintenance_report_view), name='view-maintenance-report'),
    path('wx/products/wave_data/', login_required(views.get_wave_data_analysis), name='wave-data'),
    path('wx/products/wave_data/get/', login_required(views.get_wave_data), name="get-wave-data"),
    path('wx/stations/stations_monitoring/', views.stationsmonitoring_form, name="stations-monitoring"),
    path('wx/stations/stations_monitoring/get/', views.get_stationsmonitoring_map_data),
    path('wx/stations/stations_monitoring/get/<int:id>/', views.get_stationsmonitoring_station_data),
    path('wx/stations/stations_monitoring/get/<int:station_id>/<int:variable_id>/', views.get_stationsmonitoring_chart_data),
    path('wx/maintenance_reports/equipment_inventory/', views.get_equipment_inventory, name="equipment-inventory"),
    path('wx/maintenance_reports/equipment_inventory/get/', views.get_equipment_inventory_data),
    path('wx/maintenance_reports/equipment_inventory/create/', views.create_equipment),
    # path('wx/maintenance_reports/equipment_inventory/delete/', views.delete_equipment),
    path('wx/maintenance_reports/equipment_inventory/update/', views.update_equipment),
    path('wx/maintenance_report/equipmenttype_data/update/', views.update_maintenance_report_equipment_type_data),
    path('api/available_data/', views.AvailableDataView.as_view()),
    path('api/user_info/', views.UserInfo.as_view()),
    path('api/data_export/', views.AppDataExportView.as_view()),
    path('api/intervals/', views.IntervalViewSet.as_view({'get': 'list'})),
    path('wx/reports/synop/', views.SynopView.as_view(), name='synop-capture-update'),
    path('wx/reports/synop/load/', views.synop_load),
    path('wx/reports/synop/update/', views.synop_update),
    path('wx/reports/synop/delete/', views.synop_delete),
    path('wx/reports/synop/form/', views.SynopFormView.as_view()),
    path('wx/reports/synop/form/load/', views.synop_load_form),
    path('wx/data/capture/monthly/', views.MonthlyFormView.as_view(), name='monthly-form'),
    path('wx/data/capture/monthly/load/', views.MonthlyFormLoad),
    path('wx/data/capture/monthly/update/', views.MonthlyFormUpdate),    
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.DOCUMENTS_URL, document_root=settings.DOCUMENTS_ROOT)

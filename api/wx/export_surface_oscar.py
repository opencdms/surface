import pyoscar
import shutil
import os

# client = pyoscar.OSCARClient()

# check if station exists the OSCAR platform
def check_station(station_wigos_id, client):
    try:
        # retrieve station information
        station_info = client.get_stations(wigos_id=station_wigos_id)
        
        # check if station exists
        if station_info['stationSearchResults']:
            return True

        return False
    
    except Exception as e:
        raise e


# upload xml file
def upload_xml(station_xml_path, client):
    try:
        with open(station_xml_path) as fh:
            data = fh.read()

        response = client.upload(data)

        return response
    
    except Exception as e:
        raise e


# modify xml file
def modify_xml(station_details, station_xml_path):
    try:
        # open file
        with open(station_xml_path, "r") as xml_file:
            xml_file_content = xml_file.read()

        updated_xml_file_content = xml_file_content

        # update wigos_id, station_name, begin_date, station_lat, station_long, station_elevation, file_datetime
        for key, value in station_details.items():
            updated_xml_file_content = updated_xml_file_content.replace(f'{{{{ {key} }}}}', value)

        # write updated contents back to file
        with open(station_xml_path, "w") as xml_file:
            xml_file.write(updated_xml_file_content)

    except Exception as e:
        raise e


# add/edit stations from SURFACE to OSCAR
def surface_to_oscar(station_info, api_token):
    # path to xml template
    template_xml_path = os.path.join(os.path.dirname(__file__), 'oscar_surface_xml/oscar_surface_template.xml')

    # path to xml of station
    station_xml_path = os.path.join(os.path.dirname(__file__), 'oscar_surface_xml/oscar_station.xml')

    # initializing station details
    station_details = {
        'wigos_id': station_info[0],
        'station_name': station_info[1],
        'begin_date': station_info[2],
        'station_lat': station_info[3],
        'station_long': station_info[4],
        'station_elevation': station_info[5],
        'wmo_region': station_info[6],
        'wmo_station_type': station_info[7],
        'reporting_status': station_info[8],
        'file_datetime': station_info[9],
        'date_established': station_info[10],
        'territory_name': station_info[11],
    }

    try:

        # check if station is already on the OSCAR platform
        # if check_station(station_details["wigos_id"], client):

            # create xml file to facilitate uploading to OSCAR
            shutil.copyfile(template_xml_path, station_xml_path)

            # modify xml file
            modify_xml(station_details, station_xml_path)

            # instantiate client to OSCAR DEPL (default)
            # dev_client = pyoscar.OSCARClient(api_token=api_token)

            # instantiate client to OSCAR production
            prod_client = pyoscar.OSCARClient(api_token=api_token, env='prod')

            # upload xml file to OSCAR
            result = upload_xml(station_xml_path, prod_client)
            print(f'Station {station_details["station_name"]} {station_details["wigos_id"]} - returned code {result}')

            return result

        # else:
        #     return
        
    except Exception as e:
        print(f'surface to oscar failed on Station: {station_details["station_name"]}, WIGOS ID: {station_details["wigos_id"]}! error: {e}')

        return {'code': 406, 'description': f'{e}'}
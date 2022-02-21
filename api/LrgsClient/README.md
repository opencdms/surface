# LrgsClient Installations
### Download and Install
 - Download the .jar file from [NOAA](https://dcs2.noaa.gov/)
 - Install With:
    ```sh
    java -jar filename.jar
    ```
 - On installation path select
    ```
    .../surface/api/LrgsClient
    ```
### Test the Instalation
 - Execute
 
    ```sh
    cd  .../surface/api/LrgsClient
    bin/rstat
    ```
### Working Inside Docker
 - In order to make the LrgsClient work inside the docker container, change the following files inside the directory `.../surface/api/LrgsClient/bin`:

    | Filename       | Line | Old                                 | New                          |
    | -------------- | ---- | ----------------------------------- | ---------------------------- |
    | decj           | 9    | DH=.../surface/api/LrgsClient       | DH=/surface/LrgsClient       |
    | getDcpMessages | 5    | .../surface/api/LrgsClient/bin/decj | /surface/LrgsClient/bin/decj |
    | msgaccess      | 5    | .../surface/api/LrgsClient/bin/decj | /surface/LrgsClient/bin/decj |
    | rtstat         | 5    | .../surface/api/LrgsClient/bin/decj | /surface/LrgsClient/bin/decj |

> Note: After changing these files, you will not be able to run the application outside the docker container. If you want that, just change the paths back to their old values.

# Connecting The Stations to the correct Nooa DCP

### Database Check
 -  Stations are present in 'satations' table
 -  Stations regions/districts are present in 'administrative region' table
 -  Stations watersheds are present in 'watershed' table
 -  Variables are present in 'variables' and 'station variables' tables
 -  Variable formats 'NOAA' and 'DCP_TXT' are present in 'variable formats' table for each variable

### NOAA Data
 - NOOA DCPS are present in 'noaa dcp' table
 - NOOA DCPS STATIONS are present in 'noaa dcp station' table, associated with correct NOAA DCP

> Note: If needed contact the administrators to get the correct data (i.e. correct dcps data, variables data, variables format data, etc)

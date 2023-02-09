=================
Development Plans
=================

All future plans, fixes etc are documented here

link to dev plans - https://docs.google.com/document/d/1sVr6ki4jghCtp8aIOCdf-Rb2S97S2iCAAmbTxV2ABJU/edit?usp=sharing


**Fixes needed before release**

#. Need to identify metadata parameters needed for new station to be properly created
#. Need a way to ingest file(s) with the following basic configuration parameters:country admin type, admin region, watershed, qc values, maybe even user roles and station profiles
#. Backend - Search bar for all pages especially QC range, persist and step
#. Ability to configure location of map upon login
#. Allow mouse pointer to show coordinates, distance and area on map and spatial analysis
#. Home page should load with left side bar minimised/collapsed AND filter minimised
#. Can we mouse over side bar to expand/collapse
#. Rename the following menu items
*. Monthly Capture to Daily Capture
*. PGIA Hourly Capture to Hourly Synoptic Form
*. Daily Means to Extremes and Means
*. Forget password features isn’t working again…this keeps on happening. I know it has to do with some gmail thing this needs to be resolved properly
When Creating a new Station…The new Station Page should look the same as the Metadata first page. At this time the two pages look completely different. Therefore:
Minimize the “Hydrology Information part” - hide the hydro options so that the user has to click “Hydrology Information” to see those options
ALL the station parameters should look like the metadata first page with the same options.
For now let us relax some of the “required fields” so that users can easily and quickly create a station in their country. Change the following from required to not required
WMO Program
WIGOS ID

SURFACE Development Plans

FAO Coastal Project - September 2022 - February 2023 - confirmed
B1 - Enhance the station metadata interface to allow users to query and define quality control limits for Range, Step and Persistence tests
B2 - Enhance the data inventory interface to include tabular representation of each station at element granularity and navigation in hourly, daily, monthly and yearly time scales
B3 - Enhance the station metadata management interface to allow users to track water level sensor maintenance procedures
B4 - Enhance the station map interface to allow users to monitor the operational status of automatic stations (including ocean monitoring stations)




Updates(July 2022)

Met with caribbean countries and had some very good discussion, all countries are anxiously awaiting the use of SURFACE. Practical steps moving forward looks like this…
Create user accounts for each country on cloud server
Understand each countries data formats(get sample data)
Marvin Forde from CIMH is will to assist countries in putting data together possibly putting it in one format for mass ingestion into SURFACE
Complete SURFACE user manual
Coordinate approach moving forward with CMO and WMO with clear objectives and timelines

ELLIGENCE - Funding is a major issue for Fabio at this time. Trying to finish the rest of the basic features through a coastal project. Those features are: 
QC frontend UI
Data Inventory
Station/Instrument Maintenance
Status Monitor
Form Builder
Fabio should be preparing an official proposal that will lay out these costs. Hoping that Ian can find funds to make up the difference.

OpenCDMS - Ian  
Will need to discuss with Ian how WMO and CMO and work together to implement solution
Discuss SURFACE user manual
Funding 
Data Acquisition and ingest(Marvin Forde-CIMH willing to assist)
Form Builder and other minor details - need to keep Abner 
Regional Training
I got a chance to look through the new climsoft UI. I love the layout and colours. Some of the metadata features and UI interfaces are similar to SURFACE. The data entry forms are different, SURFACE is tabular with aggregations and summations done at the bottom of each table while the new climsoft has individual blocks to enter data - - - We need to really look at the new Climsoft and SURFACE and see how to have a more coordinated approach. What are the dev plans of climsoft?


Basic cost structure for 1 developer

Part time - 4-6 hours @ $1200usd per Month
Contract - $2,000 - 3,000
Permanent - $3000 - $4000



WIS2.0 Pilot project

We also need to agree on the scope. As part of the project we would look to:
Ensure all participating stations are registered in the OSCAR/Surface database
Metadata storage and export into OSCAR
Remove any barriers to the global exchange of AWS data using the WIS 2.0 infrastructure. - should interact with DATA
Create and exchange timely daily (DAYCLI) and monthly climate summaries (CLIMAT).
 
This would all be done in the framework of the OpenCDMS and WIS2.0.



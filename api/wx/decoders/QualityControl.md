# QualityControl Range Logic
## Step 1: Current Station Thresholds
 - In order to ensure the most precise threshold the first thing we try is to set range thresholds using the correct combination (station, variable, month, and interval), from the table 'wx_qcrangethreshold'.
 
     > We also do for NULL interval to allow any interval, not needing to define thresholds for each interval used.

 - If the correct combination doesn't have thresholds set, we also search for NULL interval.
##  Step 2: Reference Station Thresholds
 - The second thing we try is to set range thresholds using the reference station thresholds. For each station, the reference station can be found at the table 'wx_station'.
 - 
    > If the station doesn't have a reference station set, we proceed to the next step.

 - If the correct combination, for the reference station, doesn't have thresholds set, we also search for NULL interval.
##  Step 3: Global Thresholds
 - The last thing we try is to set range thresholds using global thresholds for the variable, present in the table 'wx_variable'
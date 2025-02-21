WITH filtered_data AS (
    SELECT
        EXTRACT(YEAR FROM ds.day) as year
        ,EXTRACT(MONTH FROM ds.day) as month
        ,ds.station_id
        ,ds.variable_id
        ,st.name AS station
        ,so.symbol AS sampling_operation
        ,CASE so.symbol
            WHEN 'MIN' THEN ds.min_value
            WHEN 'MAX' THEN ds.max_value
            WHEN 'ACCUM' THEN ds.sum_value
            ELSE ds.avg_value
        END as value
    FROM daily_summary ds
    JOIN wx_station st ON st.id = ds.station_id
    JOIN wx_variable vr ON vr.id = ds.variable_id
    JOIN wx_samplingoperation so ON so.id = vr.sampling_operation_id   
    WHERE ds.day >= '{{start_date}}'
        AND ds.day < '{{end_date}}'
        AND ds.station_id = {{station_id}}
        AND ds.variable_id IN ({{variable_ids}})
        AND EXTRACT(MONTH FROM ds.day) IN ({{months}})                                             
)
SELECT
    station
    ,variable_id
    ,year
    ,month
    ,ROUND(
        CASE sampling_operation
            WHEN 'MIN' THEN MIN(value)::numeric
            WHEN 'MAX' THEN MAX(value)::numeric
            WHEN 'STDV' THEN STDDEV(value)::numeric
            WHEN 'ACCUM' THEN SUM(value)::numeric
            WHEN 'RMS' THEN SQRT(AVG(POW(value, 2)))::numeric
            ELSE AVG(value)::numeric
        END, 2
    ) AS value
FROM filtered_data
GROUP BY station, variable_id, month, year, sampling_operation
ORDER BY year, month
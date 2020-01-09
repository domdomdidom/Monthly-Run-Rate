SELECT
created_day, 

CASE 
    WHEN media_channel = '' THEN 'NULL'
    WHEN media_channel IS NULL THEN 'NULL'
    ELSE media_channel END AS media_channel, 

CASE 
    WHEN vertical  = '' THEN 'NULL'
    WHEN vertical  IS NULL THEN 'NULL'
    WHEN vertical IN ('Insurance, Auto', 'Auto Insurance') THEN 'Insurance, Auto'
    WHEN vertical IN ('Home Insurance', 'Insurance, Home') THEN 'Insurance, Home'
    WHEN vertical IN ('Motorcycle Insurance') THEN 'Insurance, Motorcycle'
    WHEN vertical IN ('Small Business Insurance') THEN 'Insurance, Small Business'
    WHEN vertical IN ('Insurance, Health', 'Health Insurance') THEN 'Insurance, Health'
    WHEN vertical IN ('Insurance, Life', 'Life Insurance') THEN 'Insurance, Life'
    ELSE vertical
    END AS VERTICAL, 

    SUM(coalesce(sh_click_net_revenue, 0) + coalesce(sh_cpa_revenue, 0) + coalesce(cpc_revenue, 0) + coalesce(dtsp_revenue, 0)) AS CLICK_REVENUE,
    SUM(coalesce(sh_click_qs_media_cost, 0) + coalesce(sh_cpa_media_cost, 0) + coalesce(ppc_cost, 0) + coalesce(cpc_commission, 0) + coalesce(dtsp_commission, 0)) AS CLICK_COST,
    SUM(coalesce(sh_call_net_revenue, 0) + coalesce(call_center_revenue, 0)) AS CALL_REVENUE,
    SUM(coalesce(sh_call_qs_media_cost, 0) + coalesce(call_center_commission, 0)) AS CALL_COST,
    SUM(coalesce(ld_rit_revenue, 0)) AS LEAD_REVENUE,
    SUM(coalesce(ld_commission, 0)) AS LEAD_COST

FROM bt_exec_report_historic_final

WHERE created_day >= date_add(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-32)
AND created_day < from_unixtime(unix_timestamp(), 'yyyy-MM-dd')
AND vertical LIKE '%Insurance%'

GROUP BY 
created_day,
CASE 
    WHEN media_channel = '' THEN 'NULL'
    WHEN media_channel IS NULL THEN 'NULL'
    ELSE media_channel END, 
CASE 
    WHEN vertical  = '' THEN 'NULL'
    WHEN vertical  IS NULL THEN 'NULL'
    WHEN vertical IN ('Insurance, Auto', 'Auto Insurance') THEN 'Insurance, Auto'
    WHEN vertical IN ('Home Insurance', 'Insurance, Home') THEN 'Insurance, Home'
    WHEN vertical IN ('Motorcycle Insurance') THEN 'Insurance, Motorcycle'
    WHEN vertical IN ('Small Business Insurance') THEN 'Insurance, Small Business'
    WHEN vertical IN ('Insurance, Health', 'Health Insurance') THEN 'Insurance, Health'
    WHEN vertical IN ('Insurance, Life', 'Life Insurance') THEN 'Insurance, Life'
    ELSE vertical END

ORDER BY VERTICAL, media_channel, created_day
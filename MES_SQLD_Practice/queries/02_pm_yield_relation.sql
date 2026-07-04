-- Purpose:
-- Compare average yield before and after each PM event.
--
-- SQLD concepts:
-- JOIN, GROUP BY, CASE, AVG, COUNT, HAVING
--
-- Interview-safe explanation:
-- AVG ignores NULL values. The CASE expression sends pre-PM wafers to one
-- average and post-PM wafers to another average.

SELECT
    pm.pm_id,
    pm.equipment_id,
    e.equipment_name,
    pm.pm_type,
    pm.pm_end_time,
    COUNT(CASE
        WHEN w.end_time < pm.pm_end_time THEN 1
    END) AS pre_pm_wafer_count,
    ROUND(AVG(CASE
        WHEN w.end_time < pm.pm_end_time THEN w.yield_rate
    END) * 100, 2) AS pre_pm_yield_pct,
    COUNT(CASE
        WHEN w.start_time >= pm.pm_end_time THEN 1
    END) AS post_pm_wafer_count,
    ROUND(AVG(CASE
        WHEN w.start_time >= pm.pm_end_time THEN w.yield_rate
    END) * 100, 2) AS post_pm_yield_pct,
    ROUND((
        AVG(CASE WHEN w.start_time >= pm.pm_end_time THEN w.yield_rate END)
        - AVG(CASE WHEN w.end_time < pm.pm_end_time THEN w.yield_rate END)
    ) * 100, 2) AS yield_change_pctp
FROM pm_history AS pm
INNER JOIN equipment_master AS e
    ON pm.equipment_id = e.equipment_id
LEFT JOIN wafer_process_log AS w
    ON pm.equipment_id = w.equipment_id
    AND w.end_time >= datetime(pm.pm_end_time, '-3 days')
    AND w.start_time <= datetime(pm.pm_end_time, '+3 days')
GROUP BY
    pm.pm_id,
    pm.equipment_id,
    e.equipment_name,
    pm.pm_type,
    pm.pm_end_time
HAVING
    COUNT(CASE WHEN w.end_time < pm.pm_end_time THEN 1 END) > 0
    AND COUNT(CASE WHEN w.start_time >= pm.pm_end_time THEN 1 END) > 0
ORDER BY
    yield_change_pctp DESC;

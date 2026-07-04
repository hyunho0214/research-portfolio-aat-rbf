-- Purpose:
-- Screen equipment/process combinations with low average yield.
--
-- SQLD concepts:
-- JOIN, GROUP BY, HAVING, AVG, MIN, SUM

SELECT
    e.equipment_id,
    e.equipment_name,
    w.process_step,
    COUNT(*) AS wafer_count,
    ROUND(AVG(w.yield_rate) * 100, 2) AS avg_yield_pct,
    ROUND(MIN(w.yield_rate) * 100, 2) AS min_yield_pct,
    SUM(CASE WHEN w.yield_rate < 0.93 THEN 1 ELSE 0 END) AS low_yield_wafer_count
FROM wafer_process_log AS w
INNER JOIN equipment_master AS e
    ON w.equipment_id = e.equipment_id
GROUP BY
    e.equipment_id,
    e.equipment_name,
    w.process_step
HAVING
    AVG(w.yield_rate) < 0.94
    OR SUM(CASE WHEN w.yield_rate < 0.93 THEN 1 ELSE 0 END) > 0
ORDER BY
    avg_yield_pct ASC;

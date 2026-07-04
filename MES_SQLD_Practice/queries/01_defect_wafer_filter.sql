-- Purpose:
-- Filter wafers that need engineering review.
--
-- SQLD concepts:
-- SELECT, JOIN, WHERE, CASE, ORDER BY

SELECT
    w.wafer_id,
    w.lot_id,
    w.product_code,
    w.process_step,
    w.equipment_id,
    e.equipment_name,
    ROUND(w.yield_rate * 100, 2) AS yield_pct,
    w.fail_die,
    w.fail_reason,
    CASE
        WHEN w.yield_rate < 0.90 THEN 'HIGH_RISK_REVIEW'
        WHEN w.fail_die >= 40 THEN 'CHECK_FAIL_DIE_COUNT'
        WHEN w.fail_reason IS NOT NULL THEN 'CHECK_FAIL_REASON'
        ELSE 'NORMAL_MONITORING'
    END AS review_priority
FROM wafer_process_log AS w
INNER JOIN equipment_master AS e
    ON w.equipment_id = e.equipment_id
WHERE
    w.yield_rate < 0.93
    OR w.fail_die >= 40
    OR w.fail_reason IS NOT NULL
ORDER BY
    w.yield_rate ASC,
    w.start_time ASC;

-- Purpose:
-- Check whether the sample MES tables were loaded.

SELECT 'equipment_master' AS table_name, COUNT(*) AS row_count
FROM equipment_master

UNION ALL

SELECT 'pm_history' AS table_name, COUNT(*) AS row_count
FROM pm_history

UNION ALL

SELECT 'wafer_process_log' AS table_name, COUNT(*) AS row_count
FROM wafer_process_log

UNION ALL

SELECT 'wafer_die_map' AS table_name, COUNT(*) AS row_count
FROM wafer_die_map;

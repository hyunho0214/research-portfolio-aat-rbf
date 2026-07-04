-- Purpose:
-- Create simple wafer-map defect pattern features before clustering.
--
-- SQLD concepts:
-- GROUP BY, CASE, SUM, derived table
--
-- This is not a full clustering model. It prepares easy-to-explain features
-- that could be passed to Python or a dashboard later.

SELECT
    summary.wafer_id,
    summary.defect_die_count,
    summary.edge_defect_count,
    summary.center_defect_count,
    summary.scratch_like_count,
    ROUND(summary.edge_defect_count * 1.0 / summary.defect_die_count, 2) AS edge_defect_ratio,
    CASE
        WHEN summary.scratch_like_count >= 5 THEN 'SCRATCH_LINE_CANDIDATE'
        WHEN summary.edge_defect_count >= 5 THEN 'EDGE_CLUSTER_CANDIDATE'
        WHEN summary.center_defect_count >= 3 THEN 'CENTER_CLUSTER_CANDIDATE'
        ELSE 'SCATTERED_DEFECT'
    END AS simple_cluster_label
FROM (
    SELECT
        wafer_id,
        SUM(CASE WHEN bin_code <> 'PASS' THEN 1 ELSE 0 END) AS defect_die_count,
        SUM(CASE
            WHEN bin_code <> 'PASS'
                 AND (ABS(die_x) >= 4 OR ABS(die_y) >= 4)
            THEN 1 ELSE 0
        END) AS edge_defect_count,
        SUM(CASE
            WHEN bin_code <> 'PASS'
                 AND ABS(die_x) <= 1
                 AND ABS(die_y) <= 1
            THEN 1 ELSE 0
        END) AS center_defect_count,
        SUM(CASE
            WHEN bin_code <> 'PASS'
                 AND defect_type = 'SCRATCH'
            THEN 1 ELSE 0
        END) AS scratch_like_count
    FROM wafer_die_map
    GROUP BY wafer_id
) AS summary
WHERE summary.defect_die_count > 0
ORDER BY
    summary.defect_die_count DESC,
    summary.wafer_id ASC;

INSERT INTO equipment_master (equipment_id, equipment_name, area_name, chamber_id) VALUES
('EQP_CVD_01', 'Metal CVD Tool 01', 'CVD', 'CH_A'),
('EQP_CVD_02', 'Metal CVD Tool 02', 'CVD', 'CH_B'),
('EQP_ETCH_01', 'Dry Etch Tool 01', 'ETCH', 'CH_A'),
('EQP_CMP_01', 'CMP Tool 01', 'CMP', 'CH_A');

INSERT INTO pm_history (pm_id, equipment_id, pm_start_time, pm_end_time, pm_type, engineer_id, pm_note) VALUES
('PM_001', 'EQP_CVD_01', '2026-06-05 06:00:00', '2026-06-05 08:00:00', 'FULL_PM', 'ENG_01', 'showerhead cleaning and particle check'),
('PM_002', 'EQP_ETCH_01', '2026-06-06 05:30:00', '2026-06-06 07:00:00', 'PARTIAL_PM', 'ENG_02', 'chamber wipe after alarm'),
('PM_003', 'EQP_CMP_01', '2026-06-04 07:00:00', '2026-06-04 08:20:00', 'PAD_CHANGE', 'ENG_03', 'pad replacement');

INSERT INTO wafer_process_log (
    wafer_id, lot_id, product_code, process_step, equipment_id,
    start_time, end_time, total_die, pass_die, fail_die, yield_rate, fail_reason
) VALUES
('W001', 'LOT_A01', 'PROD_A', 'METAL_CVD', 'EQP_CVD_01', '2026-06-02 10:00:00', '2026-06-02 11:20:00', 500, 443, 57, 0.886, 'PARTICLE'),
('W002', 'LOT_A01', 'PROD_A', 'METAL_CVD', 'EQP_CVD_01', '2026-06-03 09:30:00', '2026-06-03 10:50:00', 500, 451, 49, 0.902, 'PARTICLE'),
('W003', 'LOT_A02', 'PROD_A', 'METAL_CVD', 'EQP_CVD_01', '2026-06-04 14:00:00', '2026-06-04 15:15:00', 500, 445, 55, 0.890, 'CENTER_DEFECT'),
('W004', 'LOT_A02', 'PROD_A', 'METAL_CVD', 'EQP_CVD_01', '2026-06-05 13:00:00', '2026-06-05 14:10:00', 500, 476, 24, 0.952, NULL),
('W005', 'LOT_A03', 'PROD_A', 'METAL_CVD', 'EQP_CVD_01', '2026-06-06 10:00:00', '2026-06-06 11:12:00', 500, 482, 18, 0.964, NULL),
('W006', 'LOT_A03', 'PROD_A', 'METAL_CVD', 'EQP_CVD_01', '2026-06-07 16:00:00', '2026-06-07 17:10:00', 500, 479, 21, 0.958, NULL),

('W007', 'LOT_B01', 'PROD_B', 'METAL_CVD', 'EQP_CVD_02', '2026-06-02 08:30:00', '2026-06-02 09:35:00', 500, 474, 26, 0.948, NULL),
('W008', 'LOT_B01', 'PROD_B', 'METAL_CVD', 'EQP_CVD_02', '2026-06-03 12:00:00', '2026-06-03 13:05:00', 500, 472, 28, 0.944, NULL),
('W009', 'LOT_B02', 'PROD_B', 'METAL_CVD', 'EQP_CVD_02', '2026-06-04 10:30:00', '2026-06-04 11:35:00', 500, 468, 32, 0.936, 'EDGE_PARTICLE'),
('W010', 'LOT_B02', 'PROD_B', 'METAL_CVD', 'EQP_CVD_02', '2026-06-06 15:00:00', '2026-06-06 16:10:00', 500, 477, 23, 0.954, NULL),

('W011', 'LOT_C01', 'PROD_C', 'DRY_ETCH', 'EQP_ETCH_01', '2026-06-04 09:00:00', '2026-06-04 09:55:00', 500, 473, 27, 0.946, NULL),
('W012', 'LOT_C01', 'PROD_C', 'DRY_ETCH', 'EQP_ETCH_01', '2026-06-05 15:00:00', '2026-06-05 15:55:00', 500, 469, 31, 0.938, 'ETCH_PROFILE'),
('W013', 'LOT_C02', 'PROD_C', 'DRY_ETCH', 'EQP_ETCH_01', '2026-06-06 12:00:00', '2026-06-06 12:55:00', 500, 463, 37, 0.926, 'ALARM_RECOVERY'),
('W014', 'LOT_C02', 'PROD_C', 'DRY_ETCH', 'EQP_ETCH_01', '2026-06-07 11:30:00', '2026-06-07 12:25:00', 500, 459, 41, 0.918, 'SCRATCH'),
('W015', 'LOT_C03', 'PROD_C', 'DRY_ETCH', 'EQP_ETCH_01', '2026-06-08 13:40:00', '2026-06-08 14:35:00', 500, 466, 34, 0.932, 'ETCH_PROFILE'),

('W016', 'LOT_D01', 'PROD_D', 'CMP', 'EQP_CMP_01', '2026-06-02 13:00:00', '2026-06-02 13:45:00', 500, 470, 30, 0.940, NULL),
('W017', 'LOT_D01', 'PROD_D', 'CMP', 'EQP_CMP_01', '2026-06-03 13:00:00', '2026-06-03 13:48:00', 500, 462, 38, 0.924, 'EDGE_THIN'),
('W018', 'LOT_D02', 'PROD_D', 'CMP', 'EQP_CMP_01', '2026-06-04 11:00:00', '2026-06-04 11:50:00', 500, 471, 29, 0.942, NULL),
('W019', 'LOT_D02', 'PROD_D', 'CMP', 'EQP_CMP_01', '2026-06-05 10:10:00', '2026-06-05 10:58:00', 500, 478, 22, 0.956, NULL),
('W020', 'LOT_D03', 'PROD_D', 'CMP', 'EQP_CMP_01', '2026-06-06 14:10:00', '2026-06-06 14:58:00', 500, 480, 20, 0.960, NULL);

INSERT INTO wafer_die_map (wafer_id, die_x, die_y, bin_code, defect_type) VALUES
('W001', 4, 0, 'FAIL', 'PARTICLE'),
('W001', 4, 1, 'FAIL', 'PARTICLE'),
('W001', 4, -1, 'FAIL', 'PARTICLE'),
('W001', -4, 0, 'FAIL', 'PARTICLE'),
('W001', 0, 4, 'FAIL', 'PARTICLE'),
('W001', 1, 4, 'FAIL', 'PARTICLE'),
('W001', -1, 4, 'FAIL', 'PARTICLE'),
('W001', 3, 4, 'FAIL', 'PARTICLE'),

('W003', 0, 0, 'FAIL', 'CENTER_DEFECT'),
('W003', 0, 1, 'FAIL', 'CENTER_DEFECT'),
('W003', 1, 0, 'FAIL', 'CENTER_DEFECT'),
('W003', -1, 0, 'FAIL', 'CENTER_DEFECT'),
('W003', 0, -1, 'FAIL', 'CENTER_DEFECT'),

('W014', -4, -4, 'FAIL', 'SCRATCH'),
('W014', -3, -3, 'FAIL', 'SCRATCH'),
('W014', -2, -2, 'FAIL', 'SCRATCH'),
('W014', -1, -1, 'FAIL', 'SCRATCH'),
('W014', 0, 0, 'FAIL', 'SCRATCH'),
('W014', 1, 1, 'FAIL', 'SCRATCH'),
('W014', 2, 2, 'FAIL', 'SCRATCH'),
('W014', 3, 3, 'FAIL', 'SCRATCH'),
('W014', 4, 4, 'FAIL', 'SCRATCH'),

('W013', -3, 2, 'FAIL', 'RANDOM'),
('W013', 2, -2, 'FAIL', 'RANDOM'),
('W013', 1, 3, 'FAIL', 'RANDOM'),
('W013', -2, -1, 'FAIL', 'RANDOM'),

('W005', 0, 0, 'PASS', NULL),
('W005', 1, 0, 'PASS', NULL),
('W005', 0, 1, 'PASS', NULL),
('W010', 0, 0, 'PASS', NULL),
('W010', 1, 1, 'PASS', NULL);

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS wafer_die_map;
DROP TABLE IF EXISTS wafer_process_log;
DROP TABLE IF EXISTS pm_history;
DROP TABLE IF EXISTS equipment_master;

CREATE TABLE equipment_master (
    equipment_id TEXT PRIMARY KEY,
    equipment_name TEXT NOT NULL,
    area_name TEXT NOT NULL,
    chamber_id TEXT NOT NULL
);

CREATE TABLE pm_history (
    pm_id TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL,
    pm_start_time TEXT NOT NULL,
    pm_end_time TEXT NOT NULL,
    pm_type TEXT NOT NULL,
    engineer_id TEXT NOT NULL,
    pm_note TEXT,
    FOREIGN KEY (equipment_id) REFERENCES equipment_master(equipment_id)
);

CREATE TABLE wafer_process_log (
    wafer_id TEXT PRIMARY KEY,
    lot_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    process_step TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    total_die INTEGER NOT NULL,
    pass_die INTEGER NOT NULL,
    fail_die INTEGER NOT NULL,
    yield_rate REAL NOT NULL CHECK (yield_rate >= 0 AND yield_rate <= 1),
    fail_reason TEXT,
    FOREIGN KEY (equipment_id) REFERENCES equipment_master(equipment_id)
);

CREATE TABLE wafer_die_map (
    wafer_id TEXT NOT NULL,
    die_x INTEGER NOT NULL,
    die_y INTEGER NOT NULL,
    bin_code TEXT NOT NULL,
    defect_type TEXT,
    FOREIGN KEY (wafer_id) REFERENCES wafer_process_log(wafer_id)
);

CREATE INDEX idx_wafer_process_equipment
    ON wafer_process_log(equipment_id);

CREATE INDEX idx_wafer_process_step
    ON wafer_process_log(process_step);

CREATE INDEX idx_pm_history_equipment
    ON pm_history(equipment_id);

CREATE INDEX idx_die_map_wafer
    ON wafer_die_map(wafer_id);

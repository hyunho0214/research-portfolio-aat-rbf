"""Build the sample MES SQLite database and execute practice SQL queries."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"
QUERY_DIR = PROJECT_ROOT / "queries"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = DATA_DIR / "mes_practice.db"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(read_text(SQL_DIR / "schema.sql"))
        conn.executescript(read_text(SQL_DIR / "seed_data.sql"))
        conn.commit()


def rows_to_markdown(headers: list[str], rows: list[sqlite3.Row]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = ["" if row[h] is None else str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def execute_query_file(conn: sqlite3.Connection, query_path: Path) -> tuple[str, int]:
    query = read_text(query_path)
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    headers = [description[0] for description in cursor.description]

    output_base = OUTPUT_DIR / query_path.stem
    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])

    md_path.write_text(rows_to_markdown(headers, rows), encoding="utf-8")
    return query_path.name, len(rows)


def main() -> None:
    build_database()

    summary_rows = []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for query_path in sorted(QUERY_DIR.glob("*.sql")):
            query_name, row_count = execute_query_file(conn, query_path)
            summary_rows.append((query_name, row_count))

    summary_lines = ["# SQL Practice Run Summary", ""]
    summary_lines.append(f"Database: `{DB_PATH.relative_to(PROJECT_ROOT)}`")
    summary_lines.append("")
    summary_lines.append("| Query | Output rows |")
    summary_lines.append("| --- | ---: |")
    for query_name, row_count in summary_rows:
        summary_lines.append(f"| `{query_name}` | {row_count} |")
    summary_lines.append("")
    summary_lines.append("All query CSV and Markdown outputs were written to `outputs/`.")
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("MES SQLD practice run completed.")
    for query_name, row_count in summary_rows:
        print(f"- {query_name}: {row_count} rows")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

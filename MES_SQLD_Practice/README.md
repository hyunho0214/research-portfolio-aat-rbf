# SQLD MES Practice: Semiconductor Yield Monitoring

SQLD 학습 내용을 반도체 MES 상황에 연결한 작은 SQLite 실습 프로젝트입니다.
목표는 어려운 SQL을 과시하는 것이 아니라, 면접에서 직접 설명 가능한 수준의
SQL로 MES 데이터 조회, 설비 PM 이력 분석, 웨이퍼 맵 전처리 흐름을 보여주는
것입니다.

참고한 흐름은 WikiDocs의 반도체 MES/ERP SQL 파이프라인 장입니다:
https://wikidocs.net/368504

## Why This Project

반도체 MES에는 웨이퍼 진행 이력, 설비 정보, PM 이력, 수율 결과, 웨이퍼 맵
불량 좌표가 함께 쌓입니다. 이 프로젝트는 그중 가장 기본적인 세 가지 분석을
다룹니다.

1. 불량 또는 저수율 웨이퍼 필터링
2. 설비 PM 전후 수율 비교
3. 웨이퍼 맵 불량 클러스터링 전처리

## SQL Difficulty Policy

면접 방어 가능성을 우선해서 SQL 난이도를 낮게 유지했습니다.

사용한 개념:

- `SELECT`, `WHERE`, `ORDER BY`
- `JOIN`
- `GROUP BY`, `HAVING`
- `CASE WHEN`
- `COUNT`, `AVG`, `SUM`, `MIN`
- 간단한 derived table

의도적으로 피한 개념:

- 복잡한 window function
- recursive query
- 긴 CTE 체인
- 실제 운영 DB 튜닝 수준의 indexing/partitioning

## Project Structure

```text
MES_SQLD_Practice/
  sql/
    schema.sql
    seed_data.sql
  queries/
    00_table_overview.sql
    01_defect_wafer_filter.sql
    02_pm_yield_relation.sql
    03_wafer_map_defect_preprocess.sql
    04_equipment_low_yield_summary.sql
  scripts/
    run_sql_practice.py
    run_validation.ps1
  docs/
    INTERVIEW_BRIEF_KR.md
    SQLD_SCOPE.md
    REFERENCE_NOTES.md
  outputs/
    query result CSV/Markdown files
```

## Tables

| Table | Purpose |
| --- | --- |
| `equipment_master` | 설비 ID, 공정 영역, chamber 정보 |
| `pm_history` | 설비 예방정비(PM) 이력 |
| `wafer_process_log` | 웨이퍼별 공정, 설비, 수율, 불량 원인 |
| `wafer_die_map` | 웨이퍼 맵의 die 좌표와 bin/불량 유형 |

## How To Run

```powershell
cd MES_SQLD_Practice
py -3 scripts\run_sql_practice.py
```

Validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_validation.ps1
```

The runner creates a local SQLite database at `data/mes_practice.db`, executes
all SQL files in `queries/`, and writes result tables to `outputs/`.

## Query Summary

| Query | Main SQL Skill | Portfolio Meaning |
| --- | --- | --- |
| `00_table_overview.sql` | `COUNT`, `UNION ALL` | 데이터가 정상 적재됐는지 확인 |
| `01_defect_wafer_filter.sql` | `WHERE`, `JOIN`, `CASE` | 저수율/불량 웨이퍼 선별 |
| `02_pm_yield_relation.sql` | `JOIN`, `AVG`, `CASE` | PM 전후 수율 변화 비교 |
| `03_wafer_map_defect_preprocess.sql` | `GROUP BY`, `CASE` | 웨이퍼 맵 불량 패턴 전처리 |
| `04_equipment_low_yield_summary.sql` | `GROUP BY`, `HAVING` | 저수율 설비 스크리닝 |

## Current Result Snapshot

After running the validation script:

- `EQP_CVD_01` shows a positive PM-after yield change in the sample data.
- Several low-yield wafers are flagged as `HIGH_RISK_REVIEW`.
- Wafer map preprocessing labels example wafers as edge, center, scratch-line,
  or scattered defect candidates.

These are synthetic practice records. They are designed for SQL learning and
portfolio explanation, not for real fab decision-making.

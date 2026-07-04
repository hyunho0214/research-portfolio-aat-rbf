# Reference Notes

Reference page:

- WikiDocs, "9장. SQL & 데이터 파이프라인 구축 실전 (반도체 MES/ERP 연동 기초)"
  - https://wikidocs.net/368504

## How This Project Adapted The Reference

The reference explains how semiconductor MES data connects equipment logs,
metrology data, defect data, and yield data. It also gives examples such as:

- filtering defective wafers,
- grouping yield by equipment,
- comparing yield before and after PM,
- preprocessing wafer-map defect coordinates.

This project keeps the same semiconductor story but intentionally reduces SQL
difficulty:

- Uses SQLite instead of Oracle-style production SQL.
- Uses small synthetic tables instead of private fab data.
- Keeps queries focused on basic SQLD concepts.
- Avoids window functions in the required practice queries.
- Documents the limitation that PM/yield comparison is correlation screening,
  not proof of causation.

## Portfolio Message

> I practiced SQLD fundamentals through a semiconductor MES scenario. I did not
> try to write overly complex SQL. Instead, I built a small reproducible
> database and used simple joins, filters, grouping, and CASE logic to answer
> realistic manufacturing questions.

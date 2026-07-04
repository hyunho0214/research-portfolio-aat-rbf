# SQLD Scope

이 프로젝트는 SQLD 학습자 수준에서 설명 가능한 SQL만 사용합니다. 목표는
"실무형 상황을 이해하고 기본 SQL로 안전하게 조회할 수 있다"는 메시지입니다.

## 사용한 SQLD 개념

| Concept | Where | Why it matters |
| --- | --- | --- |
| `SELECT` | all queries | 필요한 컬럼만 조회 |
| `WHERE` | defect wafer filter | 조건 기반 불량 웨이퍼 선별 |
| `JOIN` | wafer + equipment, PM + wafer | MES 테이블 연결 |
| `GROUP BY` | PM/yield, equipment summary | 설비별/웨이퍼별 집계 |
| `HAVING` | low-yield summary | 집계 결과 기준 필터링 |
| `CASE WHEN` | priority, PM before/after, map labels | 업무 규칙을 SQL 컬럼으로 표현 |
| `AVG`, `COUNT`, `SUM`, `MIN` | summaries | 수율 KPI와 불량 개수 계산 |

## 의도적으로 피한 내용

- window function: 강력하지만 초심자 면접 방어 부담이 큼.
- recursive SQL: 이 프로젝트 목적과 맞지 않음.
- 복잡한 CTE 체인: 가독성보다 "어려워 보이는 SQL"이 될 위험이 있음.
- DB 운영 튜닝: partition, optimizer hint, execution plan은 별도 주제.

## 면접에서 말할 수 있는 기준

> SQLD 공부를 하면서 반도체 MES 테이블을 가정해 작은 실습 DB를 만들었습니다.
> 실제 회사 DB를 다룬 것은 아니지만, 웨이퍼 이력, 설비 PM, 수율, 웨이퍼 맵
> 데이터를 어떤 키로 연결하고 어떤 조건으로 필터링하는지 연습했습니다. 너무
> 복잡한 SQL보다 기본 `JOIN`, `GROUP BY`, `CASE`를 정확하게 쓰는 데 초점을
> 맞췄습니다.

# 면접 설명 브리프

## 한 줄 설명

SQLD에서 배운 기본 SQL을 반도체 MES 데이터 관리 상황에 적용해, 저수율 웨이퍼
필터링, 설비 PM 전후 수율 비교, 웨이퍼 맵 불량 전처리를 구현한 SQLite 실습
프로젝트입니다.

## 왜 쉽게 만들었는가

이 프로젝트는 SQL 고급 기능을 과시하기 위한 것이 아닙니다. 면접에서 직접
설명할 수 있는 수준의 SQL로 반도체 데이터가 어떻게 연결되는지 보여주는 것이
목표입니다.

그래서 `SELECT`, `WHERE`, `JOIN`, `GROUP BY`, `HAVING`, `CASE` 위주로
작성했습니다.

## 분석 1: 불량 웨이퍼 필터링

파일: `queries/01_defect_wafer_filter.sql`

설명:

- 웨이퍼 공정 로그와 설비 마스터를 `equipment_id`로 조인합니다.
- 수율이 낮거나, fail die 수가 많거나, 불량 원인이 기록된 웨이퍼를 조회합니다.
- `CASE`로 리뷰 우선순위를 붙입니다.

면접 답변 포인트:

> 모든 웨이퍼를 보는 대신, 수율 기준과 불량 원인 기준으로 먼저 engineering
> review 대상만 좁히는 쿼리입니다.

## 분석 2: 설비 PM 이력과 수율 비교

파일: `queries/02_pm_yield_relation.sql`

설명:

- PM 이력과 웨이퍼 공정 로그를 `equipment_id`로 연결합니다.
- PM 완료 시점 기준 전 3일과 후 3일의 평균 수율을 비교합니다.
- `AVG(CASE WHEN ...)` 구조로 pre-PM, post-PM 수율을 한 번에 계산합니다.

면접 답변 포인트:

> 이 결과는 "상관관계 탐색"입니다. PM이 수율을 개선했다고 단정하는 인과
> 분석은 아니고, PM 전후 수율 변화가 있었는지 확인하는 1차 스크리닝입니다.

## 분석 3: 웨이퍼 맵 불량 클러스터링 전처리

파일: `queries/03_wafer_map_defect_preprocess.sql`

설명:

- 웨이퍼 맵의 die 좌표에서 fail die만 집계합니다.
- edge defect, center defect, scratch-like defect 수를 만듭니다.
- SQL 단계에서 간단한 후보 라벨을 붙입니다.

면접 답변 포인트:

> SQL에서 복잡한 클러스터링 모델을 직접 만든 것이 아니라, Python이나 대시보드로
> 넘기기 전에 기본 feature를 만드는 전처리 단계입니다.

## 예상 질문과 답변

Q. 왜 SQLite를 썼나요?

A. 이 프로젝트는 SQLD 실습과 포트폴리오 재현성이 목적이라 누구나 실행 가능한
SQLite를 사용했습니다. 실제 회사에서는 Oracle, MSSQL, MySQL 같은 RDBMS를 쓸 수
있지만, 기본 SQL 구조는 유사합니다.

Q. PM 전후 수율 비교는 인과관계인가요?

A. 아닙니다. 이 쿼리는 PM 전후 수율 변화가 있는지 보는 상관관계/스크리닝입니다.
정확한 인과관계를 보려면 제품, 레시피, 공정 조건, lot mix를 더 통제해야 합니다.

Q. 웨이퍼 맵 클러스터링을 SQL로 한 건가요?

A. 클러스터링 모델 자체를 SQL로 구현한 것은 아닙니다. SQL에서는 edge defect
ratio, center defect count, scratch-like count 같은 전처리 feature를 만들고,
이후 Python이나 시각화 도구로 넘기는 흐름을 가정했습니다.

Q. 왜 복잡한 윈도우 함수를 쓰지 않았나요?

A. SQLD 학습 단계에서 기본 `JOIN`, `GROUP BY`, `CASE`를 정확하게 쓰는 것이 더
중요하다고 판단했습니다. 필요하면 이후 수율 trend 분석에서 `LAG()` 같은 window
function을 확장할 수 있습니다.

# FEDL Plotting Plan

## 프로젝트 목표
- plotting GUI와 MATLAB 생성 스크립트를 안정화한다.
- 그래프 본체가 설명 박스, legend, GUI 워터마크에 의해 가려지지 않도록 레이아웃을 정리한다.

## 현재 상태 / 확인 사항
- plotting/plot.m 은 삭제됨.
- 현재 MATLAB 그림은 하단 provenance 설명이 figure 내부를 크게 차지해 x축과 legend를 가린다.
- 현재 로그 스케일 선택 시 출력이 선형처럼 보이는 문제가 있다.
- GUI 워터마크는 현재 화면 하단 좌측에 배치되어 있다.

## 범위
- plotting/matlab_generator.py 레이아웃 수정
- plotting/plotting_gui.py 워터마크 위치 수정
- 생성 스크립트/파이썬 오류 검증

## 제약 및 가정
- 기존 Series / Group / Plot builder 구조는 유지한다.
- 설명 텍스트는 삭제하지 않고 그래프 바깥 별도 영역으로 이동한다.
- MATLAB R2025a 기준으로 동작시킨다.

## 산출물
- 수정된 plotting/matlab_generator.py
- 수정된 plotting/plotting_gui.py
- 검증 결과 요약

## 단계별 실행 계획
1. 현재 figure 레이아웃과 로그 축 처리 위치를 수정한다.
2. GUI 워터마크 위치를 겹치지 않게 조정한다.
3. Python 구문 검증과 스크립트 생성 검증을 수행한다.
4. 가능하면 MATLAB 재실행으로 출력 구조를 확인한다.

## 진행 로그
- 2026-03-20: plot.m 삭제 후 레이아웃/축/legend/워터마크 이슈 수정 작업 시작.
- 2026-03-20: MATLAB figure를 tiledlayout 기반으로 재구성해 plot 영역과 provenance 영역을 분리함.
- 2026-03-20: 로그 스케일을 semilogy 대신 axes YScale 속성으로 강제하도록 변경함.
- 2026-03-20: GUI 워터마크를 우측 상단으로 이동함.
- 2026-03-20: 출력 옵션을 팝업 기반으로 확장(축 범위, 주/보조 눈금 간격, 저장 포맷 포함).
- 2026-03-20: 옵션 기반 미리보기 창(Preview) 추가.
- 2026-03-20: MATLAB 축 스타일을 좌하단 중심으로 조정(상단/우측 축 비활성화).

## 사용자 변경 요청
- plot.m 은 이미 삭제됨.
- Sweep 설명을 그래프 바깥으로 이동.
- x축이 보이도록 조정.
- y축 로그 스케일이 실제 로그로 보이게 수정.
- legend 잘림 방지.
- GUI 워터마크 이동.
- X/Y 축 범위 및 주/보조 눈금 간격을 GUI에서 설정 가능하게 변경.
- 공간 이슈를 위해 출력 옵션을 팝업으로 이동.
- 출력 옵션 적용 후 예시 그림(미리보기) 제공.
- MATLAB 플롯에서 상단/우측 축 눈금 제거.

## 검증 / 테스트 상태
- plotting/plotting_gui.py: 에러 없음
- plotting/matlab_generator.py: 에러 없음
- generated_plotting_demo.m 재생성 성공
- MATLAB batch 실행 후 Plot1_V1_Abs_Id_log.png / .fig / .svg 생성 확인
- 출력 옵션 확장 후 generated_plotting_demo.m 재생성 성공

## 다음 단계
- 필요 시 provenance 글자 크기/패널 높이/legend 위치를 추가 미세조정

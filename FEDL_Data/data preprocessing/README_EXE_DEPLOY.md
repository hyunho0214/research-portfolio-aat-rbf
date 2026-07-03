# RawDataExtractor 배포 안내

## [상대방에게 전달할 때 - 간단한 방법]
✓ **dist/RawDataExtractor.exe** 파일 1개만 전달하면 됩니다!
- 상대방이 exe를 더블클릭하면 바로 실행됩니다
- Python이나 다른 파일 설치가 필요 없습니다
- 다른 파일들(build_exe.bat, raw_data_gui.py 등)은 전달할 필요가 없습니다

## [개발자가 exe를 빌드할 때]
1. 이 폴더에서 Command Prompt 또는 PowerShell을 열기
2. 다음 명령 실행: `build_exe.bat`
3. 또는 직접 빌드: `py -3 -m PyInstaller --noconfirm --clean --onefile --windowed --name RawDataExtractor raw_data_gui.py`
4. dist/RawDataExtractor.exe 파일이 생성됨

## [exe 배포 시 주의사항]
- **대상 OS**: Windows 10/11 (64비트)
- **가능한 경고**: Windows SmartScreen 또는 백신 프로그램에서 경고할 수 있음 (서명되지 않은 실행파일이므로)
- **경고가 나온 경우 해결법**:
  1. 파일을 우클릭 → 속성 → 차단 해제 클릭 → 적용
  2. 또는 관리자 권한으로 실행
- **엄격한 회사 환경**: 회사 정책상 exe 실행이 차단된 경우 IT 팀에 문의 필요

## If exe does not run on another PC
- Rebuild using the same or older Python on build PC.
- Test the exe on a clean Windows account.
- If needed, distribute the entire `dist` folder (for one-folder mode), but this project uses one-file mode.

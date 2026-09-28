# CLAUDE.md

## 프로젝트 개요
- Django 프로젝트 (AdvancedProjectLab-04)
- 루트 디렉터리: `APL_04/` (`manage.py` 위치)
- 설정 패키지: `APL_04/setting/`
- 앱: `APL_04/meetalign/`

## 명령어
`APL_04/` 디렉터리에서 실행한다.

- 개발 서버: `python manage.py runserver`
- 마이그레이션 생성: `python manage.py makemigrations`
- 마이그레이션 적용: `python manage.py migrate`
- 테스트: `python manage.py test`

## 규칙
- 모델 변경 시 마이그레이션을 함께 생성한다.
- 새 기능에는 `tests.py`에 테스트를 추가한다.
- 최대한 단순한 기능을 사용한다.
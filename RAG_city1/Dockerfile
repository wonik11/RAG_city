# 파이썬 3.10 슬림 버전 사용 (버전은 로컬 환경에 맞게 수정 가능)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 컴파일에 필요한 OS 수준의 빌드 도구 설치
RUN apt-get update && apt-get install -y gcc g++ python3-dev

# 파이썬 빌드 필수 패키지 선행 설치 및 업그레이드
RUN pip install --upgrade pip setuptools==65.5.1 wheel Cython

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 코드 복사
COPY . .

# 포트 개방
EXPOSE 8080

# FastAPI 앱 실행 (main.py 파일의 app 객체를 실행한다고 가정)
# 만약 파일명이 app.py라면 "app:app"으로 변경하세요.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

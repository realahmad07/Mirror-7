FROM python:3.11-slim
WORKDIR /app
COPY . /app
ENV PYTHONUNBUFFERED=1
ENV MIRROR7_HOST=0.0.0.0
ENV MIRROR7_PORT=8787
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('MIRROR7_PORT','8787')+'/api/health',timeout=3)"
CMD ["python","-m","mirror7_backend.http_server"]

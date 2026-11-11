FROM python:3.11-slim

# Create unprivileged service user and group
RUN groupadd -r mirror7 && useradd -r -g mirror7 -d /app -s /sbin/nologin mirror7

WORKDIR /app
COPY . /app
RUN chown -R mirror7:mirror7 /app

USER mirror7

ENV PYTHONUNBUFFERED=1
ENV MIRROR7_HOST=0.0.0.0
ENV MIRROR7_PORT=8787
ENV MIRROR7_CORS_ORIGIN=*
ENV MIRROR_LOG_LEVEL=INFO

EXPOSE 8787

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('MIRROR7_PORT','8787')+'/api/healthz',timeout=3)"

CMD ["python", "-m", "mirror7_backend.http_server"]


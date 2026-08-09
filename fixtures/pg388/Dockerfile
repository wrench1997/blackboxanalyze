# PG-388 local disposable logic lab. Supply a reviewed immutable base digest.
ARG PYTHON_IMAGE_DIGEST
FROM python:3.11-slim@${PYTHON_IMAGE_DIGEST}

WORKDIR /opt/pg388
COPY app/ ./app/
COPY fixtures/pg388/logic_lab.py ./logic_lab.py

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PG388_BIND=127.0.0.1 PG388_PORT=8088
USER nobody
EXPOSE 8088
ENTRYPOINT ["python", "logic_lab.py"]

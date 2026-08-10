# PG-388 independent local holdout implementation.  Supply a separately
# reviewed immutable Python base digest; this image is optional (compose
# profile: holdout) and is never a training authorization by itself.
ARG PYTHON_IMAGE_DIGEST_B
FROM python:3.11-slim@${PYTHON_IMAGE_DIGEST_B}

WORKDIR /opt/pg388
COPY app/ ./app/
COPY fixtures/pg388/logic_lab_b.py ./logic_lab_b.py

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PG388_BIND=0.0.0.0 PG388_PORT=8089
USER nobody
EXPOSE 8089
ENTRYPOINT ["python", "logic_lab_b.py"]

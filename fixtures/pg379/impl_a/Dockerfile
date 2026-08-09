# PG-379 implementation A.  The digest is intentionally supplied by the
# operator at build time; until it is reviewed, this image remains unbound and
# the PG-379 live gate must stay blocked.
ARG PYTHON_IMAGE_DIGEST
FROM python:3.11-slim@${PYTHON_IMAGE_DIGEST}

WORKDIR /opt/pg379_impl_a
COPY app.py build_manifest.py manifest_v1.json ./

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PG379_IMPL_A_BIND=127.0.0.1 \
    PG379_IMPL_A_PORT=8080

USER nobody
ENTRYPOINT ["python", "app.py"]

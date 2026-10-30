FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# Pinned Python dependencies (wheels only). pybanksy (GPL-3.0), harmonypy (GPL-3.0+), igraph
# (GPL-2.0+) and leidenalg (GPL-3.0) run in-process, so the image as a whole is GPL; see THIRD_PARTY.md.
COPY requirements.txt /processor/requirements.txt
RUN pip install --only-binary=:all: -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -type d -name tests -prune -exec rm -rf {} +

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# Build matplotlib's font cache and import everything once, so a run needs no writes outside /tmp.
# Every cache directory warmed here is made writable for uid 1000.
RUN mkdir -p /tmp/matplotlib /tmp/numba \
    && python -c "import matplotlib.pyplot, scanpy, banksy.main, harmonypy, leidenalg" \
    && chmod -R a+rwX /tmp/matplotlib /tmp/numba

# Runs as uid 1000 on the platform; nothing here needs root.
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]

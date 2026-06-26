FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONPATH=/opt/poultry_project

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libimage-exiftool-perl \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /tmp/matplotlib

WORKDIR /opt/poultry_project/poultry_data_preparation

COPY container_related/requirements.container.txt /tmp/requirements.container.txt

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r /tmp/requirements.container.txt \
 && rm -f /tmp/requirements.container.txt

COPY requirements.txt /opt/poultry_project/poultry_data_preparation/requirements.txt
COPY pytest.ini /opt/poultry_project/poultry_data_preparation/pytest.ini
COPY README.md /opt/poultry_project/poultry_data_preparation/README.md
COPY INPUT_FOLDER_STRUCTURE.md /opt/poultry_project/poultry_data_preparation/INPUT_FOLDER_STRUCTURE.md
COPY One_line_cmd_run_all.txt /opt/poultry_project/poultry_data_preparation/One_line_cmd_run_all.txt
COPY config /opt/poultry_project/poultry_data_preparation/config
COPY src /opt/poultry_project/poultry_data_preparation/src
COPY tests /opt/poultry_project/poultry_data_preparation/tests

WORKDIR /opt/poultry_project

CMD ["python", "-m", "poultry_data_preparation.src.main", "--help"]

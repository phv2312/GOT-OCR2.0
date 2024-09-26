FROM python:3.10-slim AS full

# Install some common dependencies
RUN apt-get update -qqy && \
    apt-get install -y \
      build-essential \
      cmake \
      g++ \
      git \
      ffmpeg \
      libsm6 \
      libxext6 \
      libxrender1 \
      locales \
      curl \
      && \
    apt-get clean && \
    apt-get autoremove

WORKDIR /app

# Install requirements
COPY requirements.lock /app/
RUN pip install -r requirements.lock --no-deps

# Remove cache
RUN rm -rf ~/.cache

# Copy entire dir
COPY . /app

# Download weights
RUN bash scripts/download_weights.sh

# Serving
CMD ["bash", "scripts/webapp_start.sh"]

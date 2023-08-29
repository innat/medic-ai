# Start with a base image of Python 3.8
FROM python:3.9

# Prevents prompts during installations
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory in the Docker image
WORKDIR /app

# Copy source code from the host to local image filesystem.
COPY . .

# Create a virtual environment and activate it
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install linux pkg and pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && apt-get purge -y --auto-remove

# Install apps pkgs
RUN python3 -m pip install -e .

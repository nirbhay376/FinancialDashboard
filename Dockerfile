FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal build deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app
COPY . /app

ENV PORT=8050

# Use shell form so $PORT expands at runtime
CMD ["sh", "-c", "gunicorn finance:server --workers 1 --worker-class gevent --bind 0.0.0.0:$PORT"]

EXPOSE 8050

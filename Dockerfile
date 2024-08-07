# Docker file for leaderboard
FROM python:3.11-slim

WORKDIR /app

# install curl
RUN apt-get update && apt-get install -y curl
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh
RUN /install.sh && rm /install.sh

# install deps
COPY leaderboard/requirements.txt ./
RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements.txt

# copy src
COPY dgeb dgeb
COPY leaderboard/ leaderboard/

# Run gradio when the container launches
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_TEMP_DIR="/app"
WORKDIR /app/leaderboard
CMD ["python", "app.py"]



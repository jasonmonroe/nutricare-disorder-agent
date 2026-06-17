# docker.sh

# -----------------+
# |     DOCKER     |
# +----------------+

# First, stop the old container if it's running
docker stop <container_id>

# Then, run the new image
docker run --rm -it -p 7860:7860 \
  --env-file .env \
  -v "$PWD":/app \
  nutricare-disorder-agent-local
  
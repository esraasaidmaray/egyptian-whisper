Deployment (Docker) quick guide

1) Build locally (CPU):
```bash
docker build -t egyptian-whisper:cpu -f Dockerfile.cpu .
```

2) Run locally:
```bash
docker run --rm -p 8000:8000 egyptian-whisper:cpu
# then open http://localhost:8000
```

3) GPU run (on machine with NVIDIA Docker support):
```bash
docker build -t egyptian-whisper:gpu -f Dockerfile.gpu .
docker run --gpus all --rm -p 8000:8000 egyptian-whisper:gpu
```

4) Push to Docker Hub (so others can pull):
```bash
docker tag egyptian-whisper:cpu YOUR_DOCKERHUB_USER/egyptian-whisper:cpu
docker push YOUR_DOCKERHUB_USER/egyptian-whisper:cpu
```

Notes:
- The image does not bundle model weights (they will download on first run). Building an image that contains the weights is possible but will create a very large image.
- For public testing you can deploy this container to a cloud VM (AWS/GCP/Hetzner) with a public IP and forward port 8000, or use a container host that supports GPUs.
- If you want an easy shareable UI, replace the FastAPI root with an embedded Gradio app (we can add that if you prefer).

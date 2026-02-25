docker build -t magic-mask-cheap-version .
docker run --gpus all -p 8003:8003 --name magic-mask-temu magic-mask-cheap-version
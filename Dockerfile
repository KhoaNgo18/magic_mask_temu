FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

WORKDIR /workspace

# Install ComfyUI
RUN git clone https://github.com/Comfy-Org/ComfyUI.git /workspace/ComfyUI

WORKDIR /workspace/ComfyUI

# Install CUDA 11.8 PyTorch wheels
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ComfyUI requirements
RUN python -m pip install -r requirements.txt
RUN python -m pip install gradio numpy

# Install custom nodes
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/kijai/ComfyUI-segment-anything-2.git && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

# Install requirements for custom nodes (if present)
RUN for d in ComfyUI-VideoHelperSuite ComfyUI-segment-anything-2 ComfyUI_essentials ComfyUI-KJNodes ComfyUI-Frame-Interpolation; do \
      if [ -f "$d/requirements.txt" ]; then \
        python -m pip install -r "$d/requirements.txt"; \
      fi; \
      if [ -f "$d/requirements-no-cupy.txt" ]; then \
        python -m pip install -r "$d/requirements-no-cupy.txt"; \
      fi; \
      if [ -f "$d/install.py" ]; then \
        (cd "$d" && python install.py); \
      fi; \
    done

WORKDIR /workspace

# Copy local Gradio app files
COPY webui_gradio.py sam_2.py frame_interpolate.py /workspace/

# Gradio network settings for container access
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8003

EXPOSE 8003

CMD ["python", "/workspace/webui_gradio.py"]

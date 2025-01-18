FROM tensorflow/tensorflow:2.14.0-gpu

# Dependencias necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir opencv-python opencv-python-headless


WORKDIR /app
COPY . /app

CMD ["bash"]

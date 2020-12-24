FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    bzip2 \
    curl \
    ca-certificates \
    git \
    libx11-6 \
    libgl1-mesa-dev \
    sudo \
    wget && \
    apt install -y build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
    mkdir opencv && \
    cd opencv && \
    git clone https://github.com/opencv/opencv.git && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv && \
    make && \
    make install && \
    # pip install --upgrade pip opencv-python opencv-contrib-python && \
    rm -rf /var/lib/apt/lists/*


# Set up UID not to use root in Container
ARG UID
ARG GID
ARG UNAME

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}

RUN groupadd -g ${GID} ${UNAME}
RUN useradd -u ${UID} -g ${UNAME} -m ${UNAME}

ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:${PATH}
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh
RUN conda install -y conda

RUN conda install pytorch==1.7.0 cudatoolkit=11.0 torchvision==0.8.1 -c pytorch && \
    conda install -c anaconda scikit-learn==0.23.2 && \
    conda install -c conda-forge matplotlib mlflow tensorboard && \
    conda clean -i -t -y

RUN pip install -U pip && \
    pip install --no-cache-dir hydra-core torchsummary opencv-python opencv-contrib-python pytorch-lightning==1.1.2

COPY bt_im.txt /tmp
COPY val.txt /tmp
COPY test.txt /tmp
WORKDIR /res

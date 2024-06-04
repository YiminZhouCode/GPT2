
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

WORKDIR /workspace

COPY . /workspace

RUN conda create -n myenv python=3.8 -y && \
    echo "conda activate myenv" >> ~/.bashrc

RUN /opt/conda/envs/myenv/bin/pip install --upgrade pip && \
    /opt/conda/envs/myenv/bin/pip install -r requirements.txt && \
    /opt/conda/envs/myenv/bin/pip install jupyter

CMD ["bash"]

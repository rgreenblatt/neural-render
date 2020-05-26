FROM greenblattryan/neural-render-generate

RUN apt-get update && apt-get install -y 
RUN apt-get update && apt-get install -y \
        curl \
        git \
        wget \
        vim \
        zip \
        build-essential \
        git

ENV PATH /opt/conda/bin:$PATH

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.7 conda-build pyyaml numpy ipython tensorboard pip && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN pip install imageio
RUN imageio_download_bin freeimage

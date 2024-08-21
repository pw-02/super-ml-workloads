# Use Alpine Linux as the base image
FROM ubuntu:22.04

ENV PYTHON_VERSION=3.10

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

# Set the working directory
WORKDIR /workspaces

RUN apt update \
    && apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   wget \
                   vim \
                   #redis-server \ 
    && rm -rf /var/lib/apt/lists

# # Copy the entrypoint script into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/conda \
    && rm miniconda3.sh \
    && /opt/conda/bin/conda install -y -c anaconda \
       python=$PYTHON_VERSION \
    && /opt/conda/bin/conda clean -ya

# RUN /opt/conda/bin/conda config --set ssl_verify False \
#     && pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
#     && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

# Install requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Set PYTHONPATH
ENV PYTHONPATH=/workspaces/super-ml-workloads

# Copy the source code for the server into the container
#COPY classification /superworkloads/classification

# Cleanup
RUN apt-get clean && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*


CMD ["/usr/local/bin/entrypoint.sh"]
#CMD ["/bin/bash"]
# Use the official CentOS image
FROM centos:latest

MAINTAINER tanchao <zmntc168@gmail.com>

# Update the repository configuration to use the vault.centos.org mirror
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

# Set the working directory inside the container
WORKDIR /app

# Update repositories and install necessary dependencies
RUN yum update -y && \
    yum install -y openssl-devel bzip2-devel libffi-devel wget sqlite-devel && \
    yum groupinstall -y "Development Tools"

# Download and install Python 3.10.2
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-loadable-sqlite-extensions --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.10  /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10  /usr/local/bin/pip3

# Install wget (if not included in the base image)
RUN yum install -y wget  wireshark && yum clean all

# install pyshark
RUN pip3 install pyshark && \
    pip3 install tqdm requests

# Copy the requrements file
COPY requirements.txt .
RUN pip3 install --no-cache-dir --no-warn-script-location -r requirements.txt

# TODO: Install additional packages for plugins

# Copy the project code
COPY taskweaver/ces /app/taskweaver/ces
COPY taskweaver/plugin /app/taskweaver/plugin
COPY taskweaver/module /app/taskweaver/module
COPY taskweaver/__init__.py /app/taskweaver/__init__.py
COPY docker/pyshark_container/entrypoint.sh /app/entrypoint.sh
RUN  chmod +x /app/entrypoint.sh

ENV PYTHONPATH "${PYTHONPATH}:/app"
#
ENTRYPOINT ["/app/entrypoint.sh"]
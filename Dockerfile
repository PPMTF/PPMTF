FROM centos:centos7.5.1804

LABEL maintainer="murakami.takao　takao-murakami at aist.go.jp"

RUN yum update -y && \
    yum install -y vim less git && \
    yum install -y wget make


# install python 3.6.5
## install packages for install python
RUN yum install -y gcc-4.8.5-44.el7.x86_64 zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel tk-devel libffi-devel

## install python 3.6.5 from source file
RUN wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tar.xz && \
    tar -xvf Python-3.6.5.tar.xz && \
    cd Python-3.6.5 && \
    ./configure && \
    make && \
    make altinstall


# clone PPMTF repository
RUN cd /opt/ && \ 
    git clone https://github.com/PPMTF/PPMTF

# install cpp libraries


## install Stats 3.3.1
RUN wget https://github.com/kthohr/stats/archive/v3.1.1.tar.gz && \
    tar -xvf ./v3.1.1.tar.gz && \
    cp -r ./stats-3.1.1/include/* /opt/PPMTF/cpp/include/

## install Eigen 3.3.7 library
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz && \
    tar -zxvf eigen-3.3.7.tar.gz && \
    cp -r ./eigen-3.3.7/Eigen /opt/PPMTF/cpp/include/


## install Gcem 1.13.1 library
RUN wget https://github.com/kthohr/gcem/archive/v1.13.1.tar.gz && \
    tar -xvf ./v1.13.1.tar.gz && \
    cp -r ./gcem-1.13.1/include/* /opt/PPMTF/cpp/include/

# install c++
RUN yum install -y gcc-c++-4.8.5-44.el7.x86_64

# install python libraries
RUN pip3.6 install scipy==1.5.4 numpy==1.19.5

# compile cpp source file
RUN cd /opt/PPMTF/cpp && \
    make

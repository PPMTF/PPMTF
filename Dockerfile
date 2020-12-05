FROM ubuntu:18.04

LABEL maintainer="murakami.takao"

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y vim less git

RUN cd /opt && \
    git clone https://github.com/PPMTF/PPMTF


## install python library
RUN pip3 install scipy numpy

## install cpp library
RUN git clone https://github.com/kthohr/stats.git && \
    cp -r ./stats/include/* /opt/PPMTF/cpp/include/

## install Eigen library
RUN git clone https://gitlab.com/libeigen/eigen.git && \
    cp -r ./eigen/Eigen /opt/PPMTF/cpp/include/


## install gcem library
RUN git clone https://github.com/kthohr/gcem.git && \
    cp -r ./gcem/include/* /opt/PPMTF/cpp/include/

RUN cd /opt/PPMTF/cpp && \
    make
   

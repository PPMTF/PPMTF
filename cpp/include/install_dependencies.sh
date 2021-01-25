#!/bin/bash -x

## install Stats 3.1.1
wget https://github.com/kthohr/stats/archive/v3.1.1.tar.gz && \
tar -zxvf v3.1.1.tar.gz && \
cp -r stats-3.1.1/include/* .

## install Eigen 3.3.7 library
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz && \
tar -zxvf eigen-3.3.7.tar.gz && \
cp -r eigen-3.3.7/Eigen .

## install Gcem 1.13.1 library
wget https://github.com/kthohr/gcem/archive/v1.13.1.tar.gz && \
tar -zxvf v1.13.1.tar.gz && \
cp -r gcem-1.13.1/include/* .

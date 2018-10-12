#!/usr/bin/env bash

VENDOR_DIR=`pwd`/vendor
CXX=clang++-6.0

# Build benchmarks

COMMON_FLAGS="-std=c++11 -O3 -DNDEBUG \
-I./vendor/include \
-L./vendor/lib \
-lbenchmark_main \
-lbenchmark \
-Wl,-rpath vendor/lib"

$CXX benchmark.cc -o benchmark-noavx $COMMON_FLAGS

$CXX benchmark.cc -o benchmark-native $COMMON_FLAGS -march=native

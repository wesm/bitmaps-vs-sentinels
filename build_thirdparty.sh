#!/usr/bin/env bash

VENDOR_DIR=`pwd`/vendor
CXX=clang++-6.0

# Install gbenchmark

function install_gbenchmark() {
    rm -rf benchmark

    git clone https://github.com/google/benchmark
    git clone https://github.com/google/googletest.git benchmark/googletest

    pushd benchmark

    cmake . -DCMAKE_BUILD_TYPE=release \
          -DCMAKE_INSTALL_PREFIX=$VENDOR_DIR
    make install

    popd
}

# Install arrow

mkdir arrow-build

pushd arrow-build

cmake ../../arrow/cpp -DCMAKE_BUILD_TYPE=release \
      -DARROW_IPC=off \
      -DARROW_BUILD_TESTS=off \
      -DARROW_WITH_SNAPPY=off \
      -DARROW_WITH_LZ4=off \
      -DARROW_WITH_ZLIB=off \
      -DARROW_WITH_ZSTD=off \
      -DARROW_WITH_BROTLI=off \
      -DCMAKE_INSTALL_PREFIX=$VENDOR_DIR

make -j8 install

popd

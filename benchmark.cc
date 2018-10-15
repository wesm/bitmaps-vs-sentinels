// Copyright 2018, Wes McKinney
// License: MIT

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <arrow/util/bit-util.h>

#include "benchmark/benchmark.h"

constexpr int64_t kArrayLength = 10000000;
constexpr double kPercentNull = 0.0;

namespace BitUtil = arrow::BitUtil;
using arrow::internal::BitmapReader;

template <typename T>
struct SumState {
  SumState() : total(0), valid_count(0) {}

  void Print() const {
    std::cout << "Total: " << this->total
              << " valid: " << this->valid_count
              << std::endl;
  }

  T total;
  int64_t valid_count;
};

template <typename T>
struct Traits {};

template <typename T>
void AddNullSentinels(double null_probability, std::vector<T>* out) {
  const int random_seed = 0;
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<double> d(0.0, 1.0);

  const T sentinel = Traits<T>::null_sentinel;

  for (size_t i = 0; i < out->size(); ++i) {
    if (d(gen) < null_probability) {
      (*out)[i] = sentinel;
    }
  }
}


template <>
struct Traits<double> {
  static constexpr double null_sentinel = static_cast<double>(NAN);

  static void Generate(int64_t length, std::vector<double>* out) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> d(-10, 10);
    out->resize(length, static_cast<double>(0));
    std::generate(out->begin(), out->end(),
                  [&d, &gen] { return static_cast<double>(d(gen)); });
  }

  static inline bool IsNull(double val) {
    return val != val;
  }

  static inline bool NotNull(double val) {
    return val == val;
  }

};

template <>
struct Traits<int64_t> {
  static constexpr int64_t null_sentinel = std::numeric_limits<int64_t>::min();

  static void Generate(int64_t length, std::vector<int64_t>* out) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<int64_t> d(-10, 10);
    out->resize(length, 0);
    std::generate(out->begin(), out->end(),
                  [&d, &gen] { return static_cast<int64_t>(d(gen)); });
  }

  static inline bool IsNull(int64_t val) {
    return val == null_sentinel;
  }

  static inline bool NotNull(int64_t val) {
    return val != null_sentinel;
  }

};

void GenerateValidBitmap(int64_t length, double null_probability,
                         std::vector<uint8_t>* out) {
  const int random_seed = 0;
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<double> d(0.0, 1.0);
  out->resize(BitUtil::BytesForBits(length), static_cast<uint8_t>(0));
  for (size_t i = 0; i < length; ++i) {
    if (d(gen) > null_probability) {
      // Set the i-th bit to 1
      BitUtil::SetBit(out->data(), i);
    }
  }
}

template <typename T>
struct SumNoNulls {
  static void Sum(const T* values, int64_t length, SumState<T>* state) {
    for (int64_t i = 0; i < length; ++i) {
      state->total += *values++;
    }
    state->valid_count += length;
  }
};

template <typename T>
struct SumNoNullsBatched {
  static void Sum(const T* values, int64_t length, SumState<T>* state) {
    const int64_t batches = length / 8;
    for (int64_t i = 0; i < batches; ++i) {
      state->total += values[0] + values[1] + values[2] + values[3]
        + values[4] + values[5] + values[6] + values[7];
      state->valid_count += 8;
      values += 8;
    }

    for (int64_t i = batches * 8; i < length; ++i) {
      state->total += *values++;
      ++state->valid_count;
    }
  }
};

template <typename T>
struct SumSentinel {
  static void Sum(const T* values, int64_t length, SumState<T>* state) {
    for (int64_t i = 0; i < length; ++i) {
      if (Traits<T>::NotNull(*values)) {
        // NaN is not equal to itself
        state->total += *values;
        ++state->valid_count;
      }
      ++values;
    }
  }
};

template <typename T>
struct SumSentinelVectorize {
  static void Sum(const T* values, int64_t length, SumState<T>* state) {
    const int64_t batches = length / 8;

#define SUM_NOT_NULL(ITEM)                      \
    do {                                        \
      if (Traits<T>::NotNull(values[ITEM])) {    \
        state->total += values[ITEM];           \
        ++state->valid_count;                   \
      }                                         \
    } while (0)

    for (int64_t i = 0; i < batches; ++i) {
      SUM_NOT_NULL(0);
      SUM_NOT_NULL(1);
      SUM_NOT_NULL(2);
      SUM_NOT_NULL(3);
      SUM_NOT_NULL(4);
      SUM_NOT_NULL(5);
      SUM_NOT_NULL(6);
      SUM_NOT_NULL(7);
      values += 8;
    }

    for (int64_t i = batches * 8; i < length; ++i) {
      if (Traits<T>::NotNull(*values)) {
        state->total += *values;
        ++state->valid_count;
      }
      ++values;
    }
  }
};

template <typename T>
struct SumBitmapNaive {
  static void Sum(const T* values, const uint8_t* valid_bitmap,
                  int64_t length, SumState<T>* state) {
    for (int64_t i = 0; i < length; ++i) {
      if (BitUtil::GetBit(valid_bitmap, i)) {
        state->total += *values;
        ++state->valid_count;
      }
      ++values;
    }
  }
};

template <typename T>
struct SumBitmapReader {
  static void Sum(const T* values, const uint8_t* valid_bitmap,
                  int64_t length, SumState<T>* state) {
    BitmapReader bit_reader(valid_bitmap, 0, length);
    for (int64_t i = 0; i < length; ++i) {
      if (bit_reader.IsSet()) {
        state->total += *values;
        ++state->valid_count;
      }
      ++values;
      bit_reader.Next();
    }
  }
};

// Generated with the following Python code

// output = 'static constexpr uint8_t kBytePopcount[] = {{{0}}};'
// popcounts = [str(bin(i).count('1')) for i in range(0, 256)]
// print(output.format(', '.join(popcounts)))

static constexpr uint8_t kBytePopcount[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

template <typename T>
struct SumBitmapVectorizeUnroll {
  static void Sum(const T* values, const uint8_t* valid_bitmap,
                  int64_t length, SumState<T>* state) {

    const int64_t whole_bytes = length / 8;
    for (int64_t i = 0; i < whole_bytes; ++i) {
      const uint8_t valid_byte = valid_bitmap[i];

      if (valid_byte < 0xFF) {
        // Some nulls
        state->total += (values[0] * (valid_byte & 1)) +
          (values[1] * ((valid_byte >> 1) & 1)) +
          (values[2] * ((valid_byte >> 2) & 1)) +
          (values[3] * ((valid_byte >> 3) & 1)) +
          (values[4] * ((valid_byte >> 4) & 1)) +
          (values[5] * ((valid_byte >> 5) & 1)) +
          (values[6] * ((valid_byte >> 6) & 1)) +
          (values[7] * ((valid_byte >> 7) & 1));
        state->valid_count += kBytePopcount[valid_byte];
      } else {
        // No nulls
        state->total += values[0] + values[1] + values[2] + values[3]
          + values[4] + values[5] + values[6] + values[7];
        state->valid_count += 8;
      }
      values += 8;
    }

    for (int64_t i = whole_bytes * 8; i < length; ++i) {
      if (BitUtil::GetBit(valid_bitmap, i)) {
        state->total = *values;
        ++state->valid_count;
      }
      ++values;
    }
  }
};

template <typename T, typename Summer>
void BenchNoNulls(benchmark::State& state) {
  std::vector<T> data;
  Traits<T>::Generate(kArrayLength, &data);

  while (state.KeepRunning()) {
    SumState<T> sum_state;
    Summer::Sum(data.data(), kArrayLength, &sum_state);
    benchmark::DoNotOptimize(sum_state);
  }
}

template <typename T, typename Summer>
void BenchSentinels(benchmark::State& state) {
  std::vector<T> data;

  Traits<T>::Generate(kArrayLength, &data);
  AddNullSentinels<T>(kPercentNull, &data);

  {
    SumState<T> sum_state;
    Summer::Sum(data.data(), kArrayLength, &sum_state);
    sum_state.Print();
  }

  while (state.KeepRunning()) {
    SumState<T> sum_state;
    Summer::Sum(data.data(), kArrayLength, &sum_state);
    benchmark::DoNotOptimize(sum_state);
  }
}

template <typename T, typename Summer>
void BenchBitmap(benchmark::State& state) {
  std::vector<T> data;
  std::vector<uint8_t> bitmap;

  Traits<T>::Generate(kArrayLength, &data);
  GenerateValidBitmap(kArrayLength, kPercentNull, &bitmap);

  {
    SumState<T> sum_state;
    Summer::Sum(data.data(), bitmap.data(), kArrayLength, &sum_state);
    sum_state.Print();
  }

  while (state.KeepRunning()) {
    SumState<T> sum_state;
    Summer::Sum(data.data(), bitmap.data(), kArrayLength, &sum_state);
    benchmark::DoNotOptimize(sum_state);
  }
}

static void BM_SumDoubleNoNulls(benchmark::State& state) {
  BenchNoNulls<double, SumNoNulls<double>>(state);
}

static void BM_SumInt64NoNulls(benchmark::State& state) {
  BenchNoNulls<int64_t, SumNoNulls<int64_t>>(state);
}

static void BM_SumDoubleNoNullsBatched(benchmark::State& state) {
  BenchNoNulls<double, SumNoNullsBatched<double>>(state);
}

static void BM_SumInt64NoNullsBatched(benchmark::State& state) {
  BenchNoNulls<int64_t, SumNoNullsBatched<int64_t>>(state);
}

static void BM_SumDoubleSentinel(benchmark::State& state) {
  BenchSentinels<double, SumSentinel<double>>(state);
}

static void BM_SumInt64Sentinel(benchmark::State& state) {
  BenchSentinels<int64_t, SumSentinel<int64_t>>(state);
}

static void BM_SumDoubleSentinelVectorize(benchmark::State& state) {
  BenchSentinels<double, SumSentinelVectorize<double>>(state);
}

static void BM_SumInt64SentinelVectorize(benchmark::State& state) {
  BenchSentinels<int64_t, SumSentinelVectorize<int64_t>>(state);
}

static void BM_SumDoubleBitmapReader(benchmark::State& state) {
  BenchBitmap<double, SumBitmapReader<double>>(state);
}

static void BM_SumInt64BitmapReader(benchmark::State& state) {
  BenchBitmap<int64_t, SumBitmapReader<int64_t>>(state);
}

static void BM_SumDoubleBitmapNaive(benchmark::State& state) {
  BenchBitmap<double, SumBitmapNaive<double>>(state);
}

static void BM_SumInt64BitmapNaive(benchmark::State& state) {
  BenchBitmap<int64_t, SumBitmapNaive<int64_t>>(state);
}

static void BM_SumDoubleBitmapVectorize(benchmark::State& state) {
  BenchBitmap<double, SumBitmapVectorizeUnroll<double>>(state);
}

static void BM_SumInt64BitmapVectorize(benchmark::State& state) {
  BenchBitmap<int64_t, SumBitmapVectorizeUnroll<int64_t>>(state);
}

BENCHMARK(BM_SumDoubleNoNulls)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64NoNulls)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SumDoubleNoNullsBatched)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64NoNullsBatched)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SumDoubleSentinel)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64Sentinel)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SumDoubleSentinelVectorize)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64SentinelVectorize)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SumDoubleBitmapNaive)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64BitmapNaive)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SumDoubleBitmapReader)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64BitmapReader)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SumDoubleBitmapVectorize)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SumInt64BitmapVectorize)->Unit(benchmark::kMicrosecond);

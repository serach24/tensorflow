/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_PLATFORM_CUS_H_
#define TENSORFLOW_CORE_PLATFORM_CUS_H_

// This type only supports conversion back and forth with float.

#include <complex>

#include "third_party/eigen3/Eigen/Core"

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define CUSTOM_DEVICE_FUNC __host__ __device__

#else
#define CUSTOM_DEVICE_FUNC

#endif

namespace tensorflow {

// https://stackoverflow.com/questions/25734477/type-casting-struct-to-integer-c

typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

struct cus {
  float value;

 public:
  void set(float f) { value = f; }
  constexpr CUSTOM_DEVICE_FUNC cus() : value(0) {}

  constexpr CUSTOM_DEVICE_FUNC cus(const float& f) : value(f) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const double& d)
      : cus(static_cast<float>(d)) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const complex64& c64)
      : cus(c64.real()) {}
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const complex128& c128)
      : cus(static_cast<float>(c128.real())) {}

  template <class T>
  explicit constexpr CUSTOM_DEVICE_FUNC cus(const T& value)
      : cus(static_cast<float>(value)) {}

  CUSTOM_DEVICE_FUNC operator float() const { return value; }

  CUSTOM_DEVICE_FUNC inline cus& operator=(float i) {
    this->set(i);
    return *this;
  }

  CUSTOM_DEVICE_FUNC inline cus& operator=(const cus& a) {
    this->set(static_cast<float>(a));
    return *this;
  }
};

CUSTOM_DEVICE_FUNC inline cus operator+(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator-(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator*(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator/(const cus& a, const cus& b) {
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

CUSTOM_DEVICE_FUNC inline cus operator+=(cus& a, const cus& b) {
  a = a + b;
  return a;
}

CUSTOM_DEVICE_FUNC inline cus operator-=(cus& a, const cus& b) {
  a = a - b;
  return a;
}

CUSTOM_DEVICE_FUNC inline cus operator*=(cus& a, const cus& b) {
  a = a * b;
  return a;
}

CUSTOM_DEVICE_FUNC inline cus operator/=(cus& a, const cus& b) {
  a = a / b;
  return a;
}

CUSTOM_DEVICE_FUNC inline bool operator<(const cus& a, const cus& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator<=(const cus& a, const cus& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator==(const cus& a, const cus& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator!=(const cus& a, const cus& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator>(const cus& a, const cus& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

CUSTOM_DEVICE_FUNC inline bool operator>=(const cus& a, const cus& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

}  // namespace tensorflow

namespace Eigen {
template <>
struct NumTraits<tensorflow::cus> : GenericNumTraits<tensorflow::cus> {
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static EIGEN_STRONG_INLINE tensorflow::cus
  epsilon() {
    return tensorflow::cus(Eigen::NumTraits<float>::epsilon());
  }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static EIGEN_STRONG_INLINE tensorflow::cus
  dummy_precision() {
    return tensorflow::cus(Eigen::NumTraits<float>::dummy_precision());
  }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static EIGEN_STRONG_INLINE tensorflow::cus
  highest() {
    return tensorflow::cus(Eigen::NumTraits<float>::highest());
  }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static EIGEN_STRONG_INLINE tensorflow::cus
  lowest() {
    return tensorflow::cus(Eigen::NumTraits<float>::lowest());
  }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static EIGEN_STRONG_INLINE tensorflow::cus
  infinity() {
    return tensorflow::cus(Eigen::NumTraits<float>::infinity());
  }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static EIGEN_STRONG_INLINE tensorflow::cus
  quiet_NaN() {
    return tensorflow::cus(Eigen::NumTraits<float>::quiet_NaN());
  }
};

}  // namespace Eigen

namespace std {
template <>
struct hash<tensorflow::cus> {
  std::size_t operator()(tensorflow::cus const& c) const noexcept {
    std::size_t h1 = std::hash<float>{}(c.value);
    return h1;
  }
};
}  // namespace std

#endif  // TENSORFLOW_CORE_PLATFORM_CUS_H_

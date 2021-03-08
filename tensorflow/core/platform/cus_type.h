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

#ifndef TENSORFLOW_CORE_PLATFORM_CUS_TYPE_H_
#define TENSORFLOW_CORE_PLATFORM_CUS_TYPE_H_

// This type only supports conversion back and forth with float.

#include<complex>
namespace tensorflow {


// https://stackoverflow.com/questions/25734477/type-casting-struct-to-integer-c


typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

struct cus_type {
  float value;

  void set(float f) { value = f; }
  cus_type(){}
  
  explicit cus_type(const float f) : value(f) {}
  explicit cus_type(const double d) : cus_type(static_cast<float>(d)) {}
  explicit cus_type(const complex64 c64) : cus_type(c64.real()) {}
  explicit cus_type(const complex128 c128) : cus_type(static_cast<float>(c128.real())) {}

  template<class T>
  explicit cus_type(const T& value) : cus_type(static_cast<float>(value)) {}


  operator float() const { return (float)value; }

  inline cus_type& operator=(float i) { 
    this->set(i);
    return *this;
  }

  inline cus_type& operator=(const cus_type& a){
    this->set(static_cast<float>(a));
    return *this;
  }
};

inline cus_type operator+(const cus_type & a, const cus_type & b){
  return cus_type(static_cast<float>(a) + static_cast<float>(b));
}

inline cus_type operator-(const cus_type & a, const cus_type & b){
  return cus_type(static_cast<float>(a) - static_cast<float>(b));
}

inline cus_type operator*(const cus_type & a, const cus_type & b){
  return cus_type(static_cast<float>(a) * static_cast<float>(b));
}

inline cus_type operator/(const cus_type & a, const cus_type & b){
  return cus_type(static_cast<float>(a) / static_cast<float>(b));
}

inline cus_type operator+=(cus_type & a, const cus_type & b){
  a = a + b;
  return a;
}

inline cus_type operator-=(cus_type & a, const cus_type & b){
  a = a - b;
  return a;
}

inline cus_type operator*=(cus_type & a, const cus_type & b){
  a = a * b;
  return a;
}

inline cus_type operator/=(cus_type & a, const cus_type & b){
  a = a / b;
  return a;
}

inline bool operator<(const cus_type & a, const cus_type & b){
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(const cus_type & a, const cus_type & b){
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator==(const cus_type & a, const cus_type & b){
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const cus_type & a, const cus_type & b){
  return static_cast<float>(a) != static_cast<float>(b);
}

inline bool operator>(const cus_type & a, const cus_type & b){
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(const cus_type & a, const cus_type & b){
  return static_cast<float>(a) >= static_cast<float>(b);
}




}  // namespace tensorflow



#endif  // TENSORFLOW_CORE_PLATFORM_CUS_TYPE_H_

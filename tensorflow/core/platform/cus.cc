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

#include "tensorflow/core/platform/cus.h"

#include <complex>
#include <cstring>

namespace tensorflow {


uint32_t cus::castF32ToValue(const float& f){
  return  *(uint32_t*) &f;
}

float cus::castValueToF32(const uint32_t& u){
  return *(float*)&u;
}

extern "C"{

cus CusAdd(cus a, cus b) {
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

cus CusSub(cus a, cus b) {
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

cus CusMul(cus a, cus b) {
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

cus CusDiv(cus a, cus b) {
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

cus CusNeg(cus a) {
  return cus(-static_cast<float>(a));
}

bool CusEq(cus a, cus b) { return static_cast<float>(a) == static_cast<float>(b); }
bool CusNe(cus a, cus b) { return static_cast<float>(a) != static_cast<float>(b); }
bool CusLt(cus a, cus b) { return static_cast<float>(a) < static_cast<float>(b);}
bool CusLe(cus a, cus b) { return static_cast<float>(a) <= static_cast<float>(b);}
bool CusGt(cus a, cus b) { return static_cast<float>(a) > static_cast<float>(b);}
bool CusGe(cus a, cus b) { return static_cast<float>(a) >= static_cast<float>(b); }

cus forceCompile() { 
  cus a;
  float b = CastCusToF32(a);
  cus c = CastF32ToCus(b);
  return c;
}

}

}  // namespace tensorflow

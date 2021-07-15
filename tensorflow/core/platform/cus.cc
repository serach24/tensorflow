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

tensorflow::cus CusAdd(tensorflow::cus a, tensorflow::cus b) {
  return cus(static_cast<float>(a) + static_cast<float>(b));
}

tensorflow::cus CusSub(tensorflow::cus a, tensorflow::cus b) {
  return cus(static_cast<float>(a) - static_cast<float>(b));
}

tensorflow::cus CusMul(tensorflow::cus a, tensorflow::cus b) {
  return cus(static_cast<float>(a) * static_cast<float>(b));
}

tensorflow::cus CusDiv(tensorflow::cus a, tensorflow::cus b) {
  return cus(static_cast<float>(a) / static_cast<float>(b));
}

tensorflow::cus CusNeg(tensorflow::cus a) {
  return cus(-static_cast<float>(a));
}

bool CusEq(tensorflow::cus a, tensorflow::cus b) { return static_cast<float>(a) == static_cast<float>(b); }
bool CusNe(tensorflow::cus a, tensorflow::cus b) { return static_cast<float>(a) != static_cast<float>(b); }
bool CusLt(tensorflow::cus a, tensorflow::cus b) { return static_cast<float>(a) < static_cast<float>(b);}
bool CusLe(tensorflow::cus a, tensorflow::cus b) { return static_cast<float>(a) <= static_cast<float>(b);}
bool CusGt(tensorflow::cus a, tensorflow::cus b) { return static_cast<float>(a) > static_cast<float>(b);}
bool CusGe(tensorflow::cus a, tensorflow::cus b) { return static_cast<float>(a) >= static_cast<float>(b); }


}  // namespace tensorflow

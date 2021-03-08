#ifndef TENSORFLOW_CORE_FRAMEWORK_CUS_TYPE_H_
#define TENSORFLOW_CORE_FRAMEWORK_CUS_TYPE_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"

// This type only supports conversion back and forth with float.

namespace tensorflow {


// https://stackoverflow.com/questions/25734477/type-casting-struct-to-integer-c

// struct cus_type {
//   float value;

//   operator float() const { return (float)value; }

//   cus_type& operator=(float i) { 
//     this->set(i);
//     return *this;
//   }

//   void set(float f) { value = f; }
//   cus_type(){}
//   explicit cus_type(const float f) : value(f) {}
// };

// Conversion routines between an array of float and cus_type of
// "size".
void FloatToCusType(const float* src, cus_type* dst, int64 size);
void CusTypeToFloat(const cus_type* src, float* dst, int64 size);

}  // namespace tensorflow



#endif  // TENSORFLOW_CORE_FRAMEWORK_CUS_TYPE_H_

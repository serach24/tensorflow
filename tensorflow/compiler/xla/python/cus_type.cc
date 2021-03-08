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

#include "tensorflow/compiler/xla/python/cus_type.h"

#include <array>
#include <locale>
// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/cus_type.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

namespace py = pybind11;

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

bool PyLong_CheckNoOverflow(PyObject* object) {
  if (!PyLong_Check(object)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

// Registered numpy type ID. Global variable populated by the registration code.
// Protected by the GIL.
int npy_cus_type = -1;

// Forward declaration.
extern PyTypeObject PyCusType_Type;

// Representation of a Python cus_type object.
struct PyCusType {
  PyObject_HEAD;  // Python object header
  cus_type value;
};

// Returns true if 'object' is a PyCusType.
bool PyCusType_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&PyCusType_Type));
}

// Extracts the value of a PyCusType object.
cus_type PyCusType_CusType(PyObject* object) {
  return reinterpret_cast<PyCusType*>(object)->value;
}

// Constructs a PyCusType object from a cus_type.
Safe_PyObjectPtr PyCusType_FromCusType(cus_type x) {
  Safe_PyObjectPtr ref =
      make_safe(PyCusType_Type.tp_alloc(&PyCusType_Type, 0));
  PyCusType* p = reinterpret_cast<PyCusType*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a cus_type value. Returns true on success,
// returns false and reports a Python error on failure.
bool CastToCusType(PyObject* arg, cus_type* output) {
  if (PyCusType_Check(arg)) {
    *output = PyCusType_CusType(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = cus_type(d);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = cus_type(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    Eigen::half f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus_type(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus_type(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus_type(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_cus_type) {
      ref = make_safe(PyArray_Cast(arr, npy_cus_type));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<cus_type*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

bool SafeCastToCusType(PyObject* arg, cus_type* output) {
  if (PyCusType_Check(arg)) {
    *output = PyCusType_CusType(arg);
    return true;
  }
  return false;
}

// Converts a PyCusType into a PyFloat.
PyObject* PyCusType_Float(PyObject* self) {
  cus_type x = PyCusType_CusType(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyCusType into a PyInt.
PyObject* PyCusType_Int(PyObject* self) {
  cus_type x = PyCusType_CusType(self);
  long y = static_cast<long>(x);  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyCusType.
PyObject* PyCusType_Negative(PyObject* self) {
  cus_type x = PyCusType_CusType(self);
  return PyCusType_FromCusType(-x).release();
}

PyObject* PyCusType_Add(PyObject* a, PyObject* b) {
  cus_type x, y;
  if (SafeCastToCusType(a, &x) && SafeCastToCusType(b, &y)) {
    return PyCusType_FromCusType(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

PyObject* PyCusType_Subtract(PyObject* a, PyObject* b) {
  cus_type x, y;
  if (SafeCastToCusType(a, &x) && SafeCastToCusType(b, &y)) {
    return PyCusType_FromCusType(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

PyObject* PyCusType_Multiply(PyObject* a, PyObject* b) {
  cus_type x, y;
  if (SafeCastToCusType(a, &x) && SafeCastToCusType(b, &y)) {
    return PyCusType_FromCusType(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

PyObject* PyCusType_TrueDivide(PyObject* a, PyObject* b) {
  cus_type x, y;
  if (SafeCastToCusType(a, &x) && SafeCastToCusType(b, &y)) {
    return PyCusType_FromCusType(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Python number methods for PyCusType objects.
PyNumberMethods PyCusType_AsNumber = {
    PyCusType_Add,       // nb_add
    PyCusType_Subtract,  // nb_subtract
    PyCusType_Multiply,  // nb_multiply
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyCusType_Negative,  // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
    PyCusType_Int,       // nb_int
    nullptr,              // reserved
    PyCusType_Float,     // nb_float

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,                // nb_floor_divide
    PyCusType_TrueDivide,  // nb_true_divide
    nullptr,                // nb_inplace_floor_divide
    nullptr,                // nb_inplace_true_divide
    nullptr,                // nb_index
};

// Constructs a new PyCusType.
PyObject* PyCusType_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to cus_type constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  cus_type value;
  if (PyCusType_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToCusType(arg, &value)) {
    return PyCusType_FromCusType(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_cus_type) {
      return PyArray_Cast(arr, npy_cus_type);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return nullptr;
}

// Comparisons on PyCusTypes.
PyObject* PyCusType_RichCompare(PyObject* a, PyObject* b, int op) {
  cus_type x, y;
  if (!SafeCastToCusType(a, &x) || !SafeCastToCusType(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      LOG(FATAL) << "Invalid op type " << op;
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyCusType.
PyObject* PyCusType_Repr(PyObject* self) {
  cus_type x = reinterpret_cast<PyCusType*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyCusType.
PyObject* PyCusType_Str(PyObject* self) {
  cus_type x = reinterpret_cast<PyCusType*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Hash function for PyCusType. We use the identity function, which is a weak
// hash function.
Py_hash_t PyCusType_Hash(PyObject* self) {
  cus_type x = reinterpret_cast<PyCusType*>(self)->value;
  return x.value;
}

// Python type for PyCusType objects.
PyTypeObject PyCusType_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "cus_type",  // tp_name
    sizeof(PyCusType),                            // tp_basicsize
    0,                                             // tp_itemsize
    nullptr,                                       // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,               // tp_getattr
    nullptr,               // tp_setattr
    nullptr,               // tp_compare / tp_reserved
    PyCusType_Repr,       // tp_repr
    &PyCusType_AsNumber,  // tp_as_number
    nullptr,               // tp_as_sequence
    nullptr,               // tp_as_mapping
    PyCusType_Hash,       // tp_hash
    nullptr,               // tp_call
    PyCusType_Str,        // tp_str
    nullptr,               // tp_getattro
    nullptr,               // tp_setattro
    nullptr,               // tp_as_buffer
                           // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "cus_type floating-point values",  // tp_doc
    nullptr,                           // tp_traverse
    nullptr,                           // tp_clear
    PyCusType_RichCompare,            // tp_richcompare
    0,                                 // tp_weaklistoffset
    nullptr,                           // tp_iter
    nullptr,                           // tp_iternext
    nullptr,                           // tp_methods
    nullptr,                           // tp_members
    nullptr,                           // tp_getset
    nullptr,                           // tp_base
    nullptr,                           // tp_dict
    nullptr,                           // tp_descr_get
    nullptr,                           // tp_descr_set
    0,                                 // tp_dictoffset
    nullptr,                           // tp_init
    nullptr,                           // tp_alloc
    PyCusType_New,                    // tp_new
    nullptr,                           // tp_free
    nullptr,                           // tp_is_gc
    nullptr,                           // tp_bases
    nullptr,                           // tp_mro
    nullptr,                           // tp_cache
    nullptr,                           // tp_subclasses
    nullptr,                           // tp_weaklist
    nullptr,                           // tp_del
    0,                                 // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyCusType_ArrFuncs;

PyArray_Descr NPyCusType_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&PyCusType_Type),
    // We must register cus_type with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != cus_type.
    // The downside of this is that NumPy scalar promotion does not work with
    // cus_type values.
    /*kind=*/'V',
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(cus_type),
    /*alignment=*/alignof(cus_type),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyCusType_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

PyObject* NPyCusType_GetItem(void* data, void* arr) {
  cus_type x;
  memcpy(&x, data, sizeof(cus_type));
  return PyCusType_FromCusType(x).release();
}

int NPyCusType_SetItem(PyObject* item, void* data, void* arr) {
  cus_type x;
  if (!CastToCusType(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 item->ob_type->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(cus_type));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

int NPyCusType_Compare(const void* a, const void* b, void* arr) {
  cus_type x;
  memcpy(&x, a, sizeof(cus_type));

  cus_type y;
  memcpy(&y, b, sizeof(cus_type));

  if (x < y) {
    return -1;
  }
  if (y < x) {
    return 1;
  }
  // NaNs sort to the end.
  if (!Eigen::numext::isnan(x) && Eigen::numext::isnan(y)) {
    return -1;
  }
  if (Eigen::numext::isnan(x) && !Eigen::numext::isnan(y)) {
    return 1;
  }
  return 0;
}

void NPyCusType_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                           npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyCusType_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyCusType_NonZero(void* data, void* arr) {
  cus_type x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<cus_type>(0);
}

int NPyCusType_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  cus_type* const buffer = reinterpret_cast<cus_type*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<cus_type>(start + i * delta);
  }
  return 0;
}

void NPyCusType_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                         void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    cus_type* const b1 = reinterpret_cast<cus_type*>(c1);
    cus_type* const b2 = reinterpret_cast<cus_type*>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  cus_type* out = reinterpret_cast<cus_type*>(op);
  *out = static_cast<cus_type>(acc);
}

int NPyCusType_CompareFunc(const void* v1, const void* v2, void* arr) {
  cus_type b1 = *reinterpret_cast<const cus_type*>(v1);
  cus_type b2 = *reinterpret_cast<const cus_type*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

int NPyCusType_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind,
                           void* arr) {
  const cus_type* bdata = reinterpret_cast<const cus_type*>(data);
  float max_val = -std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) > max_val) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

int NPyCusType_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind,
                           void* arr) {
  const cus_type* bdata = reinterpret_cast<const cus_type*>(data);
  float min_val = std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) < min_val) {
      min_val = static_cast<float>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

// NumPy casts

template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
};

template <>
struct TypeDescriptor<cus_type> {
  typedef cus_type T;
  static int Dtype() { return npy_cus_type; }
};

template <>
struct TypeDescriptor<uint8> {
  typedef uint8 T;
  static int Dtype() { return NPY_UINT8; }
};

template <>
struct TypeDescriptor<uint16> {
  typedef uint16 T;
  static int Dtype() { return NPY_UINT16; }
};

template <>
struct TypeDescriptor<uint32> {
  typedef uint32 T;
  static int Dtype() { return NPY_UINT32; }
};

template <typename Uint64Type>
struct TypeDescriptor<
    Uint64Type, typename std::enable_if<std::is_integral<Uint64Type>::value &&
                                        !std::is_signed<Uint64Type>::value &&
                                        sizeof(Uint64Type) == 8>::type> {
  typedef Uint64Type T;
  static int Dtype() { return NPY_UINT64; }
};

template <>
struct TypeDescriptor<int8> {
  typedef int8 T;
  static int Dtype() { return NPY_INT8; }
};

template <>
struct TypeDescriptor<int16> {
  typedef int16 T;
  static int Dtype() { return NPY_INT16; }
};

template <>
struct TypeDescriptor<int32> {
  typedef int32 T;
  static int Dtype() { return NPY_INT32; }
};

template <typename Int64Type>
struct TypeDescriptor<
    Int64Type, typename std::enable_if<std::is_integral<Int64Type>::value &&
                                       std::is_signed<Int64Type>::value &&
                                       sizeof(Int64Type) == 8>::type> {
  typedef Int64Type T;
  static int Dtype() { return NPY_INT64; }
};

template <>
struct TypeDescriptor<bool> {
  typedef int8 T;
  static int Dtype() { return NPY_BOOL; }
};

template <>
struct TypeDescriptor<Eigen::half> {
  typedef Eigen::half T;
  static int Dtype() { return NPY_HALF; }
};

template <>
struct TypeDescriptor<float> {
  typedef float T;
  static int Dtype() { return NPY_FLOAT; }
};

template <>
struct TypeDescriptor<double> {
  typedef double T;
  static int Dtype() { return NPY_DOUBLE; }
};

template <>
struct TypeDescriptor<complex64> {
  typedef complex64 T;
  static int Dtype() { return NPY_COMPLEX64; }
};

template <>
struct TypeDescriptor<complex128> {
  typedef complex128 T;
  static int Dtype() { return NPY_COMPLEX128; }
};

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] =
        static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));
  }
}

// Registers a cast between cus_type and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that cus_type can be
// safely coerced to T.
template <typename T>
bool RegisterCusTypeCast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_cus_type,
                               NPyCast<T, cus_type>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyCusType_Descr, numpy_type,
                               NPyCast<cus_type, T>) < 0) {
    return false;
  }
  if (cast_is_safe && PyArray_RegisterCanCast(&NPyCusType_Descr, numpy_type,
                                              NPY_NOSCALAR) < 0) {
    return false;
  }
  return true;
}

template <typename InType, typename OutType, typename Functor>
struct UnaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    char* o = args[1];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) = Functor()(x);
      i0 += steps[0];
      o += steps[1];
    }
  }
};

template <typename InType, typename OutType, typename OutType2,
          typename Functor>
struct UnaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype(),
            TypeDescriptor<OutType2>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    char* o0 = args[1];
    char* o1 = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      std::tie(*reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o0),
               *reinterpret_cast<typename TypeDescriptor<OutType2>::T*>(o1)) =
          Functor()(x);
      i0 += steps[0];
      o0 += steps[1];
      o1 += steps[2];
    }
  }
};

template <typename InType, typename OutType, typename Functor>
struct BinaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      auto y = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
  }
};

template <typename InType, typename InType2, typename OutType, typename Functor>
struct BinaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType2>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      auto y =
          *reinterpret_cast<const typename TypeDescriptor<InType2>::T*>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
  }
};

template <typename UFunc>
bool RegisterUFunc(PyObject* numpy, const char* name) {
  std::vector<int> types = UFunc::Types();
  PyUFuncGenericFunction fn =
      reinterpret_cast<PyUFuncGenericFunction>(UFunc::Call);
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %lu", name,
                 ufunc->nargs, types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, npy_cus_type, fn,
                                  const_cast<int*>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

struct Add {
  cus_type operator()(cus_type a, cus_type b) { return a + b; }
};
struct Subtract {
  cus_type operator()(cus_type a, cus_type b) { return a - b; }
};
struct Multiply {
  cus_type operator()(cus_type a, cus_type b) { return a * b; }
};
struct TrueDivide {
  cus_type operator()(cus_type a, cus_type b) { return a / b; }
};

std::pair<float, float> divmod(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan};
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = std::copysign(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = std::copysign(0.0f, a / b);
  }
  return {floordiv, mod};
}

struct FloorDivide {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(divmod(static_cast<float>(a), static_cast<float>(b)).first);
  }
};
struct Remainder {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(
        divmod(static_cast<float>(a), static_cast<float>(b)).second);
  }
};
struct DivmodUFunc {
  static std::vector<int> Types() {
    return {npy_cus_type, npy_cus_type, npy_cus_type, npy_cus_type};
  }
  static void Call(char** args, npy_intp* dimensions, npy_intp* steps,
                   void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o0 = args[2];
    char* o1 = args[3];
    for (npy_intp k = 0; k < *dimensions; k++) {
      cus_type x = *reinterpret_cast<const cus_type*>(i0);
      cus_type y = *reinterpret_cast<const cus_type*>(i1);
      float floordiv, mod;
      std::tie(floordiv, mod) =
          divmod(static_cast<float>(x), static_cast<float>(y));
      *reinterpret_cast<cus_type*>(o0) = cus_type(floordiv);
      *reinterpret_cast<cus_type*>(o1) = cus_type(mod);
      i0 += steps[0];
      i1 += steps[1];
      o0 += steps[2];
      o1 += steps[3];
    }
  }
};
struct Fmod {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(std::fmod(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Negative {
  cus_type operator()(cus_type a) { return -a; }
};
struct Positive {
  cus_type operator()(cus_type a) { return a; }
};
struct Power {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(std::pow(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Abs {
  cus_type operator()(cus_type a) {
    return cus_type(std::abs(static_cast<float>(a)));
  }
};
struct Cbrt {
  cus_type operator()(cus_type a) {
    return cus_type(std::cbrt(static_cast<float>(a)));
  }
};
struct Ceil {
  cus_type operator()(cus_type a) {
    return cus_type(std::ceil(static_cast<float>(a)));
  }
};
struct CopySign {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(
        std::copysign(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Exp {
  cus_type operator()(cus_type a) {
    return cus_type(std::exp(static_cast<float>(a)));
  }
};
struct Exp2 {
  cus_type operator()(cus_type a) {
    return cus_type(std::exp2(static_cast<float>(a)));
  }
};
struct Expm1 {
  cus_type operator()(cus_type a) {
    return cus_type(std::expm1(static_cast<float>(a)));
  }
};
struct Floor {
  cus_type operator()(cus_type a) {
    return cus_type(std::floor(static_cast<float>(a)));
  }
};
struct Frexp {
  std::pair<cus_type, int> operator()(cus_type a) {
    int exp;
    float f = std::frexp(static_cast<float>(a), &exp);
    return {cus_type(f), exp};
  }
};
struct Heaviside {
  cus_type operator()(cus_type bx, cus_type h0) {
    float x = static_cast<float>(bx);
    if (Eigen::numext::isnan(x)) {
      return bx;
    }
    if (x < 0) {
      return cus_type(0.0f);
    }
    if (x > 0) {
      return cus_type(1.0f);
    }
    return h0;  // x == 0
  }
};
struct Conjugate {
  cus_type operator()(cus_type a) { return a; }
};
struct IsFinite {
  bool operator()(cus_type a) { return std::isfinite(static_cast<float>(a)); }
};
struct IsInf {
  bool operator()(cus_type a) { return std::isinf(static_cast<float>(a)); }
};
struct IsNan {
  bool operator()(cus_type a) {
    return Eigen::numext::isnan(static_cast<float>(a));
  }
};
struct Ldexp {
  cus_type operator()(cus_type a, int exp) {
    return cus_type(std::ldexp(static_cast<float>(a), exp));
  }
};
struct Log {
  cus_type operator()(cus_type a) {
    return cus_type(std::log(static_cast<float>(a)));
  }
};
struct Log2 {
  cus_type operator()(cus_type a) {
    return cus_type(std::log2(static_cast<float>(a)));
  }
};
struct Log10 {
  cus_type operator()(cus_type a) {
    return cus_type(std::log10(static_cast<float>(a)));
  }
};
struct Log1p {
  cus_type operator()(cus_type a) {
    return cus_type(std::log1p(static_cast<float>(a)));
  }
};
struct LogAddExp {
  cus_type operator()(cus_type bx, cus_type by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return cus_type(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return cus_type(out);
  }
};
struct LogAddExp2 {
  cus_type operator()(cus_type bx, cus_type by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return cus_type(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return cus_type(out);
  }
};
struct Modf {
  std::pair<cus_type, cus_type> operator()(cus_type a) {
    float integral;
    float f = std::modf(static_cast<float>(a), &integral);
    return {cus_type(f), cus_type(integral)};
  }
};

struct Reciprocal {
  cus_type operator()(cus_type a) {
    return cus_type(1.f / static_cast<float>(a));
  }
};
struct Rint {
  cus_type operator()(cus_type a) {
    return cus_type(std::rint(static_cast<float>(a)));
  }
};
struct Sign {
  cus_type operator()(cus_type a) {
    float f(a);
    if (f < 0) {
      return cus_type(-1);
    }
    if (f > 0) {
      return cus_type(1);
    }
    return a;
  }
};
struct SignBit {
  bool operator()(cus_type a) { return std::signbit(static_cast<float>(a)); }
};
struct Sqrt {
  cus_type operator()(cus_type a) {
    return cus_type(std::sqrt(static_cast<float>(a)));
  }
};
struct Square {
  cus_type operator()(cus_type a) {
    float f(a);
    return cus_type(f * f);
  }
};
struct Trunc {
  cus_type operator()(cus_type a) {
    return cus_type(std::trunc(static_cast<float>(a)));
  }
};

// Trigonometric functions
struct Sin {
  cus_type operator()(cus_type a) {
    return cus_type(std::sin(static_cast<float>(a)));
  }
};
struct Cos {
  cus_type operator()(cus_type a) {
    return cus_type(std::cos(static_cast<float>(a)));
  }
};
struct Tan {
  cus_type operator()(cus_type a) {
    return cus_type(std::tan(static_cast<float>(a)));
  }
};
struct Arcsin {
  cus_type operator()(cus_type a) {
    return cus_type(std::asin(static_cast<float>(a)));
  }
};
struct Arccos {
  cus_type operator()(cus_type a) {
    return cus_type(std::acos(static_cast<float>(a)));
  }
};
struct Arctan {
  cus_type operator()(cus_type a) {
    return cus_type(std::atan(static_cast<float>(a)));
  }
};
struct Arctan2 {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(std::atan2(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Hypot {
  cus_type operator()(cus_type a, cus_type b) {
    return cus_type(std::hypot(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Sinh {
  cus_type operator()(cus_type a) {
    return cus_type(std::sinh(static_cast<float>(a)));
  }
};
struct Cosh {
  cus_type operator()(cus_type a) {
    return cus_type(std::cosh(static_cast<float>(a)));
  }
};
struct Tanh {
  cus_type operator()(cus_type a) {
    return cus_type(std::tanh(static_cast<float>(a)));
  }
};
struct Arcsinh {
  cus_type operator()(cus_type a) {
    return cus_type(std::asinh(static_cast<float>(a)));
  }
};
struct Arccosh {
  cus_type operator()(cus_type a) {
    return cus_type(std::acosh(static_cast<float>(a)));
  }
};
struct Arctanh {
  cus_type operator()(cus_type a) {
    return cus_type(std::atanh(static_cast<float>(a)));
  }
};
struct Deg2rad {
  cus_type operator()(cus_type a) {
    static constexpr float radians_per_degree = M_PI / 180.0f;
    return cus_type(static_cast<float>(a) * radians_per_degree);
  }
};
struct Rad2deg {
  cus_type operator()(cus_type a) {
    static constexpr float degrees_per_radian = 180.0f / M_PI;
    return cus_type(static_cast<float>(a) * degrees_per_radian);
  }
};

struct Eq {
  npy_bool operator()(cus_type a, cus_type b) { return a == b; }
};
struct Ne {
  npy_bool operator()(cus_type a, cus_type b) { return a != b; }
};
struct Lt {
  npy_bool operator()(cus_type a, cus_type b) { return a < b; }
};
struct Gt {
  npy_bool operator()(cus_type a, cus_type b) { return a > b; }
};
struct Le {
  npy_bool operator()(cus_type a, cus_type b) { return a <= b; }
};
struct Ge {
  npy_bool operator()(cus_type a, cus_type b) { return a >= b; }
};
struct Maximum {
  cus_type operator()(cus_type a, cus_type b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa > fb ? a : b;
  }
};
struct Minimum {
  cus_type operator()(cus_type a, cus_type b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa < fb ? a : b;
  }
};
struct Fmax {
  cus_type operator()(cus_type a, cus_type b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa > fb ? a : b;
  }
};
struct Fmin {
  cus_type operator()(cus_type a, cus_type b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa < fb ? a : b;
  }
};

struct LogicalNot {
  npy_bool operator()(cus_type a) { return !a; }
};
struct LogicalAnd {
  npy_bool operator()(cus_type a, cus_type b) { return a && b; }
};
struct LogicalOr {
  npy_bool operator()(cus_type a, cus_type b) { return a || b; }
};
struct LogicalXor {
  npy_bool operator()(cus_type a, cus_type b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

struct NextAfter {
  cus_type operator()(cus_type from, cus_type to) {
    uint16_t from_as_int, to_as_int;
    const uint16_t sign_mask = 1 << 15;
    float from_as_float(from), to_as_float(to);
    memcpy(&from_as_int, &from, sizeof(cus_type));
    memcpy(&to_as_int, &to, sizeof(cus_type));
    if (Eigen::numext::isnan(from_as_float) ||
        Eigen::numext::isnan(to_as_float)) {
      return cus_type(std::numeric_limits<float>::quiet_NaN());
    }
    if (from_as_int == to_as_int) {
      return to;
    }
    if (from_as_float == 0) {
      if (to_as_float == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        uint16_t out_int = (to_as_int & sign_mask) | 1;
        cus_type out;
        memcpy(&out, &out_int, sizeof(cus_type));
        return out;
      }
    }
    uint16_t from_sign = from_as_int & sign_mask;
    uint16_t to_sign = to_as_int & sign_mask;
    uint16_t from_abs = from_as_int & ~sign_mask;
    uint16_t to_abs = to_as_int & ~sign_mask;
    uint16_t magnitude_adjustment =
        (from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
    uint16_t out_int = from_as_int + magnitude_adjustment;
    cus_type out;
    memcpy(&out, &out_int, sizeof(cus_type));
    return out;
  }
};

// TODO(phawkins): implement spacing

}  // namespace ufuncs

}  // namespace

// Initializes the module.
bool Initialize() {
  import_array1(false);
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  PyCusType_Type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&PyCusType_Type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyCusType_ArrFuncs);
  NPyCusType_ArrFuncs.getitem = NPyCusType_GetItem;
  NPyCusType_ArrFuncs.setitem = NPyCusType_SetItem;
  NPyCusType_ArrFuncs.compare = NPyCusType_Compare;
  NPyCusType_ArrFuncs.copyswapn = NPyCusType_CopySwapN;
  NPyCusType_ArrFuncs.copyswap = NPyCusType_CopySwap;
  NPyCusType_ArrFuncs.nonzero = NPyCusType_NonZero;
  NPyCusType_ArrFuncs.fill = NPyCusType_Fill;
  NPyCusType_ArrFuncs.dotfunc = NPyCusType_DotFunc;
  NPyCusType_ArrFuncs.compare = NPyCusType_CompareFunc;
  NPyCusType_ArrFuncs.argmax = NPyCusType_ArgMaxFunc;
  NPyCusType_ArrFuncs.argmin = NPyCusType_ArgMinFunc;

  Py_TYPE(&NPyCusType_Descr) = &PyArrayDescr_Type;
  npy_cus_type = PyArray_RegisterDataType(&NPyCusType_Descr);
  if (npy_cus_type < 0) {
    return false;
  }

  // Support dtype(cus_type)
  if (PyDict_SetItemString(PyCusType_Type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyCusType_Descr)) <
      0) {
    return false;
  }

  // Register casts
  if (!RegisterCusTypeCast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusTypeCast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusTypeCast<bool>(NPY_BOOL, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<uint8>(NPY_UINT8, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<uint16>(NPY_UINT16, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<uint32>(NPY_UINT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<uint64>(NPY_UINT64, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<int8>(NPY_INT8, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<int16>(NPY_INT16, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<int32>(NPY_INT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<int64>(NPY_INT64, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<npy_longlong>(NPY_LONGLONG,
                                          /*cast_is_safe=*/false)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCusTypeCast<complex64>(NPY_COMPLEX64, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusTypeCast<complex128>(NPY_COMPLEX128,
                                        /*cast_is_safe=*/true)) {
    return false;
  }

  bool ok =
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Add>>(numpy.get(),
                                                                  "add") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Subtract>>(
          numpy.get(), "subtract") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Multiply>>(
          numpy.get(), "multiply") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::TrueDivide>>(
          numpy.get(), "divide") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::LogAddExp>>(
          numpy.get(), "logaddexp") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::LogAddExp2>>(
          numpy.get(), "logaddexp2") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Negative>>(
          numpy.get(), "negative") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Positive>>(
          numpy.get(), "positive") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::TrueDivide>>(
          numpy.get(), "true_divide") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::FloorDivide>>(
          numpy.get(), "floor_divide") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Power>>(numpy.get(),
                                                                    "power") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Remainder>>(
          numpy.get(), "remainder") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Remainder>>(
          numpy.get(), "mod") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Fmod>>(numpy.get(),
                                                                   "fmod") &&
      RegisterUFunc<ufuncs::DivmodUFunc>(numpy.get(), "divmod") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Abs>>(numpy.get(),
                                                                 "absolute") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Abs>>(numpy.get(),
                                                                 "fabs") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Rint>>(numpy.get(),
                                                                  "rint") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Sign>>(numpy.get(),
                                                                  "sign") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Heaviside>>(
          numpy.get(), "heaviside") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Conjugate>>(
          numpy.get(), "conjugate") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Exp>>(numpy.get(),
                                                                 "exp") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Exp2>>(numpy.get(),
                                                                  "exp2") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Expm1>>(numpy.get(),
                                                                   "expm1") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Log>>(numpy.get(),
                                                                 "log") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Log2>>(numpy.get(),
                                                                  "log2") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Log10>>(numpy.get(),
                                                                   "log10") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Log1p>>(numpy.get(),
                                                                   "log1p") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Sqrt>>(numpy.get(),
                                                                  "sqrt") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Square>>(numpy.get(),
                                                                    "square") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Cbrt>>(numpy.get(),
                                                                  "cbrt") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Reciprocal>>(
          numpy.get(), "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Sin>>(numpy.get(),
                                                                 "sin") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Cos>>(numpy.get(),
                                                                 "cos") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Tan>>(numpy.get(),
                                                                 "tan") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Arcsin>>(numpy.get(),
                                                                    "arcsin") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Arccos>>(numpy.get(),
                                                                    "arccos") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Arctan>>(numpy.get(),
                                                                    "arctan") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Arctan2>>(
          numpy.get(), "arctan2") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Hypot>>(numpy.get(),
                                                                    "hypot") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Sinh>>(numpy.get(),
                                                                  "sinh") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Cosh>>(numpy.get(),
                                                                  "cosh") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Tanh>>(numpy.get(),
                                                                  "tanh") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Arcsinh>>(
          numpy.get(), "arcsinh") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Arccosh>>(
          numpy.get(), "arccosh") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Arctanh>>(
          numpy.get(), "arctanh") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Deg2rad>>(
          numpy.get(), "deg2rad") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Rad2deg>>(
          numpy.get(), "rad2deg") &&

      // Comparison functions
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::Eq>>(numpy.get(),
                                                             "equal") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::Ne>>(numpy.get(),
                                                             "not_equal") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::Lt>>(numpy.get(),
                                                             "less") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::Gt>>(numpy.get(),
                                                             "greater") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::Le>>(numpy.get(),
                                                             "less_equal") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::Ge>>(numpy.get(),
                                                             "greater_equal") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Maximum>>(
          numpy.get(), "maximum") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Minimum>>(
          numpy.get(), "minimum") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Fmax>>(numpy.get(),
                                                                   "fmax") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::Fmin>>(numpy.get(),
                                                                   "fmin") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::LogicalAnd>>(
          numpy.get(), "logical_and") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::LogicalOr>>(
          numpy.get(), "logical_or") &&
      RegisterUFunc<BinaryUFunc<cus_type, bool, ufuncs::LogicalXor>>(
          numpy.get(), "logical_xor") &&
      RegisterUFunc<UnaryUFunc<cus_type, bool, ufuncs::LogicalNot>>(
          numpy.get(), "logical_not") &&

      // Floating point functions
      RegisterUFunc<UnaryUFunc<cus_type, bool, ufuncs::IsFinite>>(numpy.get(),
                                                                  "isfinite") &&
      RegisterUFunc<UnaryUFunc<cus_type, bool, ufuncs::IsInf>>(numpy.get(),
                                                               "isinf") &&
      RegisterUFunc<UnaryUFunc<cus_type, bool, ufuncs::IsNan>>(numpy.get(),
                                                               "isnan") &&
      RegisterUFunc<UnaryUFunc<cus_type, bool, ufuncs::SignBit>>(numpy.get(),
                                                                 "signbit") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::CopySign>>(
          numpy.get(), "copysign") &&
      RegisterUFunc<UnaryUFunc2<cus_type, cus_type, cus_type, ufuncs::Modf>>(
          numpy.get(), "modf") &&
      RegisterUFunc<BinaryUFunc2<cus_type, int, cus_type, ufuncs::Ldexp>>(
          numpy.get(), "ldexp") &&
      RegisterUFunc<UnaryUFunc2<cus_type, cus_type, int, ufuncs::Frexp>>(
          numpy.get(), "frexp") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Floor>>(numpy.get(),
                                                                   "floor") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Ceil>>(numpy.get(),
                                                                  "ceil") &&
      RegisterUFunc<UnaryUFunc<cus_type, cus_type, ufuncs::Trunc>>(numpy.get(),
                                                                   "trunc") &&
      RegisterUFunc<BinaryUFunc<cus_type, cus_type, ufuncs::NextAfter>>(
          numpy.get(), "nextafter");

  return ok;
}

StatusOr<py::object> CusTypeDtype() {
  if (npy_cus_type < 0) {
    // Not yet initialized. We assume the GIL protects npy_cus_type.
    if (!Initialize()) {
      return InternalError("CusType numpy type initialization failed.");
    }
  }
  return py::object(reinterpret_cast<PyObject*>(&PyCusType_Type),
                    /*is_borrowed=*/true);
}

}  // namespace xla

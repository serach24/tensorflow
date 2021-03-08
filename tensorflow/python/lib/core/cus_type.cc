#include <array>

#include "tensorflow/python/lib/core/cus_type.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

// Workarounds for Python 2 vs 3 API differences.
#if PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyString_FromString(s.c_str());
}

typedef long HashType;  // NOLINT

bool TfPyInt_Check(PyObject* object) { return PyInt_Check(object); }

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyInt_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyInt_AsLong(x);
}

#else  // PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyUnicode_FromString(s.c_str());
}

bool TfPyInt_Check(PyObject* object) {
  if (!PyLong_Check(object)) {
    return 0;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyLong_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyLong_AsLong(x);
}

typedef Py_hash_t HashType;

#endif  // PY_MAJOR_VERSION < 3

// Forward declaration.
extern PyTypeObject PyCusType_Type;

// Representation of a Python custom object.
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
bool AsCusType(PyObject* arg, cus_type* output) {
  if (PyCusType_Check(arg)) {
    *output = PyCusType_CusType(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    *output = cus_type(d);
    return true;
  }
  if (TfPyInt_Check(arg)) {
    long l = TfPyInt_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = cus_type(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus_type(f);
    return true;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
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
  return TfPyInt_FromLong(y);
}

// Negates a PyCusType.
PyObject* PyCusType_Negative(PyObject* self) {
  cus_type x = PyCusType_CusType(self);
  x.value = -x.value;
  return PyCusType_FromCusType(x).release();
}

// Binary arithmetic operators on PyCusType values.
#define CUSTOM_BINOP(name, op)                                  \
  PyObject* PyCusType_##name(PyObject* a, PyObject* b) {         \
    cus_type x, y;                                                \
    if (!AsCusType(a, &x) || !AsCusType(b, &y)) return nullptr; \
    cus_type z = x op y;                                          \
    return PyCusType_FromCusType(z).release();                  \
  }
CUSTOM_BINOP(Add, +)
CUSTOM_BINOP(Subtract, -)
CUSTOM_BINOP(Multiply, *)
CUSTOM_BINOP(Divide, /)
#undef CUSTOM_BINOP

// Python number methods for PyCusType objects.
PyNumberMethods PyCusType_AsNumber = {
    PyCusType_Add,       // nb_add
    PyCusType_Subtract,  // nb_subtract
    PyCusType_Multiply,  // nb_multiply
#if PY_MAJOR_VERSION < 3
    PyCusType_Divide,  // nb_divide
#endif
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
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_coerce
#endif
    PyCusType_Int,  // nb_int
#if PY_MAJOR_VERSION < 3
    PyCusType_Int,  // nb_long
#else
    nullptr,  // reserved
#endif
    PyCusType_Float,  // nb_float
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_oct
    nullptr,  // nb_hex
#endif

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_inplace_divide
#endif
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,            // nb_floor_divide
    PyCusType_Divide,  // nb_true_divide
    nullptr,            // nb_inplace_floor_divide
    nullptr,            // nb_inplace_true_divide
    nullptr,            // nb_index
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

  if (PyCusType_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else {
    cus_type value;
    if (!AsCusType(arg, &value)) {
      return nullptr;
    }
    return PyCusType_FromCusType(value).release();
  }
}

// Comparisons on PyCusTypes.
PyObject* PyCusType_RichCompare(PyObject* a, PyObject* b, int op) {
  cus_type x, y;
  if (!AsCusType(a, &x) || !AsCusType(b, &y)) return nullptr;
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
  string v = strings::StrCat("cus_type(", static_cast<float>(x), ")");
  return MakePyString(v);
}

// Implementation of str() for PyCusType.
PyObject* PyCusType_Str(PyObject* self) {
  cus_type x = reinterpret_cast<PyCusType*>(self)->value;
  string v = strings::StrCat(static_cast<float>(x));
  return MakePyString(v);
}

// Hash function for PyCusType. We use the identity function, which is a weak
// hash function.
HashType PyCusType_Hash(PyObject* self) {
  cus_type x = reinterpret_cast<PyCusType*>(self)->value;
  return x.value;
}

// Python type for PyCusType objects.
PyTypeObject PyCusType_Type = {
#if PY_MAJOR_VERSION < 3
    PyObject_HEAD_INIT(nullptr) 0,  // ob_size
#else
    PyVarObject_HEAD_INIT(nullptr, 0)
#endif
    "cus_type",                                // tp_name
    sizeof(PyCusType),                        // tp_basicsize
    0,                                         // tp_itemsize
    nullptr,                                   // tp_dealloc
    nullptr,                                   // tp_print
    nullptr,                                   // tp_getattr
    nullptr,                                   // tp_setattr
    nullptr,                                   // tp_compare / tp_reserved
    PyCusType_Repr,                           // tp_repr
    &PyCusType_AsNumber,                      // tp_as_number
    nullptr,                                   // tp_as_sequence
    nullptr,                                   // tp_as_mapping
    PyCusType_Hash,                           // tp_hash
    nullptr,                                   // tp_call
    PyCusType_Str,                            // tp_str
    nullptr,                                   // tp_getattro
    nullptr,                                   // tp_setattro
    nullptr,                                   // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
    "cus_type floating-point values",          // tp_doc
    nullptr,                                   // tp_traverse
    nullptr,                                   // tp_clear
    PyCusType_RichCompare,                    // tp_richcompare
    0,                                         // tp_weaklistoffset
    nullptr,                                   // tp_iter
    nullptr,                                   // tp_iternext
    nullptr,                                   // tp_methods
    nullptr,                                   // tp_members
    nullptr,                                   // tp_getset
    nullptr,                                   // tp_base
    nullptr,                                   // tp_dict
    nullptr,                                   // tp_descr_get
    nullptr,                                   // tp_descr_set
    0,                                         // tp_dictoffset
    nullptr,                                   // tp_init
    nullptr,                                   // tp_alloc
    PyCusType_New,                            // tp_new
    nullptr,                                   // tp_free
    nullptr,                                   // tp_is_gc
    nullptr,                                   // tp_bases
    nullptr,                                   // tp_mro
    nullptr,                                   // tp_cache
    nullptr,                                   // tp_subclasses
    nullptr,                                   // tp_weaklist
    nullptr,                                   // tp_del
    0,                                         // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyCusType_ArrFuncs;

PyArray_Descr NPyCusType_Descr = {
    PyObject_HEAD_INIT(nullptr) & PyCusType_Type,  // typeobj
    // We must register cus_type with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != cus_type.
    'V',  // kind
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    'E',                                                  // type
    '=',                                                  // byteorder
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,  // hasobject
    0,                                                    // type_num
    sizeof(cus_type),                                     // elsize
    alignof(cus_type),                                    // alignment
    nullptr,                                              // subarray
    nullptr,                                              // fields
    nullptr,                                              // names
    &NPyCusType_ArrFuncs,                                // f
};

// Registered numpy type ID. Global variable populated by the registration code.
int npy_cus_type_ = -1;

// Implementations of NumPy array methods.

PyObject* NPyCusType_GetItem(void* data, void* arr) {
  cus_type x;
  memcpy(&x, data, sizeof(cus_type));
  return PyCusType_FromCusType(x).release();
}

int NPyCusType_SetItem(PyObject* item, void* data, void* arr) {
  cus_type x;
  if (!AsCusType(item, &x)) return -1;
  memcpy(data, &x, sizeof(cus_type));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
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

// NumPy casts

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const From* from = reinterpret_cast<From*>(from_void);
  To* to = reinterpret_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(from[i]);
  }
}

// Registers a cast between cus_type and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that cus_type can be
// safely coerced to T.
template <typename T>
bool RegisterCusTypeCast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_cus_type_,
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
void BinaryUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                 void* data) {
  const char* i0 = args[0];
  const char* i1 = args[1];
  char* o = args[2];
  for (npy_intp k = 0; k < *dimensions; k++) {
    InType x = *reinterpret_cast<const InType*>(i0);
    InType y = *reinterpret_cast<const InType*>(i1);
    *reinterpret_cast<OutType*>(o) = Functor()(x, y);
    i0 += steps[0];
    i1 += steps[1];
    o += steps[2];
  }
}

template <typename Functor>
void CompareUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                  void* data) {
  BinaryUFunc<cus_type, npy_bool, Functor>(args, dimensions, steps, data);
}

struct CusTypeEqFunctor {
  npy_bool operator()(cus_type a, cus_type b) { return a == b; }
};
struct CusTypeNeFunctor {
  npy_bool operator()(cus_type a, cus_type b) { return a != b; }
};
struct CusTypeLtFunctor {
  npy_bool operator()(cus_type a, cus_type b) { return a < b; }
};
struct CusTypeGtFunctor {
  npy_bool operator()(cus_type a, cus_type b) { return a > b; }
};
struct CusTypeLeFunctor {
  npy_bool operator()(cus_type a, cus_type b) { return a <= b; }
};
struct CusTypeGeFunctor {
  npy_bool operator()(cus_type a, cus_type b) { return a >= b; }
};

// Initializes the module.
bool Initialize() {
  // It's critical to import umath to avoid crash in open source build.
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(MakePyString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // We hit a mysterious crash if we haven't initialized numpy before this:
  PyCusType_Type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&PyCusType_Type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyCusType_ArrFuncs);
  NPyCusType_ArrFuncs.getitem = NPyCusType_GetItem;
  NPyCusType_ArrFuncs.setitem = NPyCusType_SetItem;
  NPyCusType_ArrFuncs.copyswapn = NPyCusType_CopySwapN;
  NPyCusType_ArrFuncs.copyswap = NPyCusType_CopySwap;
  NPyCusType_ArrFuncs.nonzero = NPyCusType_NonZero;
  NPyCusType_ArrFuncs.fill = NPyCusType_Fill;

  Py_TYPE(&NPyCusType_Descr) = &PyArrayDescr_Type;
  npy_cus_type_ = PyArray_RegisterDataType(&NPyCusType_Descr);
  if (npy_cus_type_ < 0) return false;

  // Support dtype(cus_type)
  if (PyDict_SetItemString(PyCusType_Type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyCusType_Descr)) <
      0) {
    return false;
  }

  // Register casts

  // We lie shamelessly and say that a cast from half to cus_type is safe.
  // Numpy frequently uses the smallest legal representation type for small
  // float constants (e.g., 1.0), which is often float16. Things break if these
  // cannot be converted transparently to cus_type.
  if (!RegisterCusTypeCast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/true)) {
    return false;
  }

  if (!RegisterCusTypeCast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusTypeCast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusTypeCast<int32>(NPY_INT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusTypeCast<int64>(NPY_INT64, /*cast_is_safe=*/false)) {
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

  // Register ufuncs
  auto register_ufunc = [&](const char* name, PyUFuncGenericFunction fn,
                            const std::array<int, 3>& types) {
    Safe_PyObjectPtr ufunc_obj =
        make_safe(PyObject_GetAttrString(numpy.get(), name));
    if (!ufunc_obj) {
      return false;
    }
    PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
    if (types.size() != ufunc->nargs) {
      PyErr_Format(PyExc_AssertionError,
                   "ufunc %s takes %d arguments, loop takes %lu", name,
                   ufunc->nargs, types.size());
      return false;
    }
    if (PyUFunc_RegisterLoopForType(ufunc, npy_cus_type_, fn,
                                    const_cast<int*>(types.data()),
                                    nullptr) < 0) {
      return false;
    }
    return true;
  };

  // Comparisons
  const std::array<int, 3> compare_types = {
      {npy_cus_type_, npy_cus_type_, NPY_BOOL}};

  if (!register_ufunc("equal", CompareUFunc<CusTypeEqFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("not_equal", CompareUFunc<CusTypeNeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less", CompareUFunc<CusTypeLtFunctor>, compare_types)) {
    return false;
  }
  if (!register_ufunc("greater", CompareUFunc<CusTypeGtFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less_equal", CompareUFunc<CusTypeLeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("greater_equal", CompareUFunc<CusTypeGeFunctor>,
                      compare_types)) {
    return false;
  }
  return true;
}

}  // namespace

void RegisterNumpyCusType() {
  if (npy_cus_type_ >= 0) {
    // Already initialized.
    return;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load cus_type module.");
    }
    PyErr_Print();
  }
}

PyObject* CusTypePyType() {
  CHECK(PyCusType_Type.tp_base != nullptr);
  Py_INCREF(&PyCusType_Type);
  return reinterpret_cast<PyObject*>(&PyCusType_Type);
}

int CusTypeNumpyType() {
  CHECK_GE(npy_cus_type_, 0);
  return npy_cus_type_;
}

}  // namespace tensorflow
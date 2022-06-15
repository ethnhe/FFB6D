#pragma once

// @generated from tools/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack());
  });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    result.push_back(v.unpack());
  }
  return result;
}

struct TypeAndSize {
  TypeAndSize() : options(at::TensorOptions()) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes().vec())
    , options(t.options()) {}

  Tensor zeros() { return at::zeros(sizes, options); }

private:
  std::vector<int64_t> sizes;
  at::TensorOptions options;
};

struct TORCH_API AbsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AbsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AcosBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcosBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }
  void release_variables() override {


  }

  at::ScalarType other_scalar_type;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;

};
struct TORCH_API AddBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;

};
struct TORCH_API AddbmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddbmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch2_.reset_data();
    batch1_.reset_data();
  }

  int64_t batch1_argsize_0 = 0;
  int64_t batch1_argsize_1 = 0;
  int64_t batch2_argsize_2 = 0;
  SavedVariable batch2_;
  at::Scalar alpha;
  SavedVariable batch1_;
  at::Scalar beta;

};
struct TORCH_API AddcdivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcdivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor2_.reset_data();
    tensor1_.reset_data();
  }

  at::ScalarType self_scalar_type;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::Scalar value;
  SavedVariable tensor1_;
  at::ScalarType tensor2_scalar_type;

};
struct TORCH_API AddcmulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcmulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor2_.reset_data();
    tensor1_.reset_data();
  }

  at::ScalarType self_scalar_type;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::Scalar value;
  SavedVariable tensor1_;
  at::ScalarType tensor2_scalar_type;

};
struct TORCH_API AddmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat2_.reset_data();
    mat1_.reset_data();
  }

  std::vector<int64_t> mat1_sizes;
  std::vector<int64_t> mat1_strides;
  SavedVariable mat2_;
  at::Scalar alpha;
  SavedVariable mat1_;
  std::vector<int64_t> mat2_sizes;
  std::vector<int64_t> mat2_strides;
  at::Scalar beta;

};
struct TORCH_API SparseAddmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseAddmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    sparse_.reset_data();
    dense_.reset_data();
  }

  SavedVariable sparse_;
  std::vector<int64_t> dense_sizes;
  std::vector<int64_t> dense_strides;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable dense_;

};
struct TORCH_API AddmvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec_.reset_data();
    mat_.reset_data();
  }

  SavedVariable vec_;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat_;

};
struct TORCH_API AddrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec2_.reset_data();
    vec1_.reset_data();
  }

  at::Scalar beta;
  SavedVariable vec2_;
  at::Scalar alpha;
  SavedVariable vec1_;

};
struct TORCH_API AffineGridGeneratorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AffineGridGeneratorBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> size;
  bool align_corners;

};
struct TORCH_API AliasBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AliasBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API AngleBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AngleBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AnyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AnyBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API AnyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AnyBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AllBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AllBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API AllBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AllBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AcoshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcoshBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AcoshBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcoshBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AsinhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AsinhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinhBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AtanhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanhBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AsStridedBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward0"; }
  void release_variables() override {


  }

  at::TensorGeometry self_geometry;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  c10::optional<int64_t> storage_offset;

};
struct TORCH_API AsinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AtanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API Atan2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Atan2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API BaddbmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BaddbmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch2_.reset_data();
    batch1_.reset_data();
  }

  SavedVariable batch2_;
  at::Scalar alpha;
  SavedVariable batch1_;
  at::Scalar beta;

};
struct TORCH_API BernoulliBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API BernoulliBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize p_info;

};
struct TORCH_API BernoulliBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward2"; }
  void release_variables() override {


  }



};
struct TORCH_API BmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    mat2_.reset_data();
  }

  SavedVariable self_;
  SavedVariable mat2_;

};
struct TORCH_API CatBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CatBackward0"; }
  void release_variables() override {


  }

  ::std::vector<::std::vector<int64_t>> tensors_args_sizes;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  int64_t dim = 0;
  size_t tensors_size_;
};
struct TORCH_API CauchyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CauchyBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API CeilBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeilBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API CholeskyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  bool upper;
  SavedVariable result_;

};
struct TORCH_API LinalgCholeskyExBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgCholeskyExBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    L_.reset_data();
  }

  bool upper;
  SavedVariable L_;

};
struct TORCH_API CholeskySolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskySolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    input2_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable input2_;
  bool upper;
  SavedVariable result_;

};
struct TORCH_API CholeskyInverseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyInverseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  bool upper;
  SavedVariable result_;

};
struct TORCH_API ClampBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    min_.reset_data();
    max_.reset_data();
  }

  SavedVariable self_;
  SavedVariable min_;
  SavedVariable max_;

};
struct TORCH_API ClampBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> min;
  c10::optional<at::Scalar> max;

};
struct TORCH_API ClampMinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar min;

};
struct TORCH_API ClampMinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    min_.reset_data();
  }

  SavedVariable self_;
  SavedVariable min_;

};
struct TORCH_API ClampMaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar max;

};
struct TORCH_API ClampMaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    max_.reset_data();
  }

  SavedVariable self_;
  SavedVariable max_;

};
struct TORCH_API CloneBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CloneBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ToCopyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToCopyBackward0"; }
  void release_variables() override {


  }

  at::TensorOptions self_options;
  bool non_blocking;

};
struct TORCH_API CoalesceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoalesceBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ComplexBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ComplexBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    imag_.reset_data();
    real_.reset_data();
  }

  SavedVariable imag_;
  SavedVariable real_;

};
struct TORCH_API PolarBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolarBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API ConjBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API NegViewBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegViewBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ConjPhysicalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjPhysicalBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ConjPhysicalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjPhysicalBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API CopysignBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CopysignBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  torch::autograd::generated::TypeAndSize other_info;
  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API CopysignBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CopysignBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API CosBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CosBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API CoshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoshBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API CrossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CrossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  c10::optional<int64_t> dim;
  SavedVariable other_;

};
struct TORCH_API LogcumsumexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogcumsumexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API CumprodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumprodBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  at::ScalarType self_scalar_type;
  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API CumsumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumsumBackward0"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  int64_t dim = 0;

};
struct TORCH_API CummaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CummaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API CumminBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumminBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API ConvTbcBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvTbcBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
    bias_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  SavedVariable bias_;
  int64_t pad = 0;

};
struct TORCH_API CtcLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CtcLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    log_probs_.reset_data();
    targets_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  SavedVariable log_probs_;
  SavedVariable targets_;
  std::vector<int64_t> input_lengths;
  std::vector<int64_t> target_lengths;
  int64_t blank = 0;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API Deg2RadBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Deg2RadBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API DetLuBasedHelperBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DetLuBasedHelperBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    det_.reset_data();
    lu_.reset_data();
    pivs_.reset_data();
  }

  SavedVariable self_;
  SavedVariable det_;
  SavedVariable lu_;
  SavedVariable pivs_;

};
struct TORCH_API DiagBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t diagonal = 0;

};
struct TORCH_API DiagonalBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t offset = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 0;

};
struct TORCH_API DiagonalBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackwardBackward0"; }
  void release_variables() override {


  }

  int64_t offset = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 0;

};
struct TORCH_API DistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;
  at::Scalar p;
  SavedVariable result_;

};
struct TORCH_API DivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;
  at::ScalarType self_scalar_type;

};
struct TORCH_API DivBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar other;

};
struct TORCH_API DivBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;
  c10::optional<std::string> rounding_mode;
  at::ScalarType self_scalar_type;

};
struct TORCH_API DivBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward3"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar other;
  c10::optional<std::string> rounding_mode;

};
struct TORCH_API DotBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DotBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor_.reset_data();
    self_.reset_data();
  }

  SavedVariable tensor_;
  SavedVariable self_;

};
struct TORCH_API VdotBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VdotBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API FusedDropoutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FusedDropoutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  double p;
  SavedVariable result1_;

};
struct TORCH_API EigBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EigBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    eigenvalues_.reset_data();
    eigenvectors_return_.reset_data();
  }

  SavedVariable self_;
  bool eigenvectors;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_return_;

};
struct TORCH_API EqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API EqBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ErfBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API ErfcBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfcBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API SpecialErfcxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialErfcxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API ErfinvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfinvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API ExpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API Exp2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Exp2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API Expm1Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Expm1Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API ExpandBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpandBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API ExponentialBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExponentialBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FakeQuantizePerTensorAffineCachemaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerTensorAffineCachemaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API FakeQuantizeLearnablePerTensorAffineBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizeLearnablePerTensorAffineBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    scale_.reset_data();
    zero_point_.reset_data();
  }

  SavedVariable self_;
  SavedVariable scale_;
  SavedVariable zero_point_;
  int64_t quant_min = 0;
  int64_t quant_max = 0;
  double grad_factor;

};
struct TORCH_API FakeQuantizePerChannelAffineCachemaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerChannelAffineCachemaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API FakeQuantizeLearnablePerChannelAffineBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizeLearnablePerChannelAffineBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    scale_.reset_data();
    zero_point_.reset_data();
  }

  SavedVariable self_;
  SavedVariable scale_;
  SavedVariable zero_point_;
  int64_t axis = 0;
  int64_t quant_min = 0;
  int64_t quant_max = 0;
  double grad_factor;

};
struct TORCH_API FusedMovingAvgObsFqHelperBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FusedMovingAvgObsFqHelperBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API FillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API FloorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FloorBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FmodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FmodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  SavedVariable other_;

};
struct TORCH_API FracBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FracBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FrexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FrexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
  }

  SavedVariable exponent_;

};
struct TORCH_API GatherBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GatherBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    index_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable index_;
  bool sparse_grad;

};
struct TORCH_API GeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GeometricBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeometricBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API GeqrfBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeqrfBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API GridSampler2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    grid_.reset_data();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
  bool align_corners;

};
struct TORCH_API GridSampler3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    grid_.reset_data();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
  bool align_corners;

};
struct TORCH_API GridSampler2DCpuFallbackBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DCpuFallbackBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    grid_.reset_data();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
  bool align_corners;

};
struct TORCH_API GtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API HardsigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardsigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API HistcBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HistcBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API HardswishBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardswishBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API HypotBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HypotBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable other_;
  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API I0Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "I0Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API SpecialI0EBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialI0EBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API SpecialI1Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialI1Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API SpecialI1EBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialI1EBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API IgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API IgammacBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IgammacBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API IndexBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;

};
struct TORCH_API IndexAddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexAddBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;
  int64_t source_dim = 0;
  SavedVariable source_;
  at::Scalar alpha;

};
struct TORCH_API IndexCopyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexCopyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;
  int64_t source_dim = 0;
  SavedVariable source_;

};
struct TORCH_API IndexFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API IndexFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API IndexPutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
  bool accumulate;

};
struct TORCH_API IndexPutImplBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutImplBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
  bool accumulate;

};
struct TORCH_API IndexSelectBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexSelectBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API InverseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "InverseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API LinalgInvExBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgInvExBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    inverse_.reset_data();
  }

  SavedVariable inverse_;

};
struct TORCH_API KthvalueBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KthvalueBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API LeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LerpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward0"; }
  void release_variables() override {


  }

  at::Scalar weight;

};
struct TORCH_API LerpBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    weight_.reset_data();
    self_.reset_data();
    end_.reset_data();
  }

  SavedVariable weight_;
  SavedVariable self_;
  SavedVariable end_;

};
struct TORCH_API LgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API DigammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DigammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API PolygammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  int64_t n = 0;
  SavedVariable self_;

};
struct TORCH_API PolygammaBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  int64_t n = 0;

};
struct TORCH_API LogBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API Log10Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log10Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API Log1PBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log1PBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API Log2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API LogaddexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogaddexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API Logaddexp2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Logaddexp2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API XlogyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  torch::autograd::generated::TypeAndSize other_info;
  SavedVariable other_;

};
struct TORCH_API XlogyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  at::Scalar self;
  torch::autograd::generated::TypeAndSize other_info;
  SavedVariable other_;

};
struct TORCH_API XlogyBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar other;

};
struct TORCH_API SpecialXlog1PyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API SpecialXlog1PyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  at::Scalar self;
  SavedVariable other_;

};
struct TORCH_API SpecialXlog1PyBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward2"; }
  void release_variables() override {


  }

  at::Scalar other;

};
struct TORCH_API SpecialZetaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialZetaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API SpecialZetaBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialZetaBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  at::Scalar self;
  SavedVariable other_;

};
struct TORCH_API SpecialZetaBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialZetaBackward2"; }
  void release_variables() override {


  }



};
struct TORCH_API LogdetBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogdetBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API LogNormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogNormalBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API LogsumexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogsumexpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API LstsqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LstsqBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API LinalgLstsqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLstsqBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API LtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LuWithInfoBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuWithInfoBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
  }

  SavedVariable self_;
  SavedVariable LU_;
  SavedVariable pivots_;

};
struct TORCH_API LuSolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuSolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    LU_data_.reset_data();
    LU_pivots_.reset_data();
  }

  SavedVariable self_;
  SavedVariable LU_data_;
  SavedVariable LU_pivots_;

};
struct TORCH_API LuUnpackBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuUnpackBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_data_.reset_data();
  }

  SavedVariable LU_data_;
  bool unpack_data;

};
struct TORCH_API MaskedFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API MaskedFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API MaskedScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedScatterBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;
  std::vector<int64_t> source_sizes;

};
struct TORCH_API MaskedSelectBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedSelectBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    mask_.reset_data();
  }

  SavedVariable self_;
  SavedVariable mask_;

};
struct TORCH_API MatrixExpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MatrixExpBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API MaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API MaximumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaximumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API FmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API MeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t self_numel = 0;
  at::ScalarType self_scalar_type;

};
struct TORCH_API MeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::ScalarType self_scalar_type;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API MedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API NanmedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanmedianBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API MedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API NanmedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanmedianBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API MinimumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinimumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API FminBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FminBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API AmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API AminBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AminBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API MmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    mat2_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> mat2_sizes;
  std::vector<int64_t> mat2_strides;
  std::vector<int64_t> self_sizes;
  std::vector<int64_t> self_strides;
  SavedVariable mat2_;

};
struct TORCH_API ModeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ModeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;
  SavedVariable other_;

};
struct TORCH_API MulBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar other;

};
struct TORCH_API MvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec_.reset_data();
    self_.reset_data();
  }

  SavedVariable vec_;
  SavedVariable self_;

};
struct TORCH_API MvlgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvlgammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  int64_t p = 0;

};
struct TORCH_API NanToNumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanToNumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API NativeBatchNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double eps;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NativeBatchNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_invstd_.reset_data();
  }

  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_invstd_;
  bool train;
  double eps;

};
struct TORCH_API NativeLayerNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    bias_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  SavedVariable input_;
  std::vector<int64_t> normalized_shape;
  SavedVariable weight_;
  SavedVariable bias_;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NativeLayerNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    mean_.reset_data();
    rstd_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_out_;
  SavedVariable input_;
  std::vector<int64_t> normalized_shape;
  SavedVariable mean_;
  SavedVariable rstd_;
  SavedVariable weight_;

};
struct TORCH_API NativeGroupNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeGroupNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  int64_t N = 0;
  int64_t C = 0;
  int64_t HxW = 0;
  int64_t group = 0;
  double eps;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API NeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API NegBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API NextafterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NextafterBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API NormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  at::Scalar p;
  SavedVariable result_;

};
struct TORCH_API NormBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> p;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API NormBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> p;
  SavedVariable result_;

};
struct TORCH_API NormBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward3"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> p;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API LinalgVectorNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgVectorNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  at::Scalar ord;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API PdistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  double p;
  SavedVariable result_;

};
struct TORCH_API PdistBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackwardBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API EuclideanDistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EuclideanDistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x2_.reset_data();
    result_.reset_data();
  }

  SavedVariable x1_;
  SavedVariable x2_;
  SavedVariable result_;

};
struct TORCH_API CdistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x2_.reset_data();
    result_.reset_data();
  }

  SavedVariable x1_;
  SavedVariable x2_;
  double p;
  SavedVariable result_;

};
struct TORCH_API CdistBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackwardBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API NormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API NormalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> mean_sizes;

};
struct TORCH_API NormalBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward2"; }
  void release_variables() override {


  }

  std::vector<int64_t> std_sizes;

};
struct TORCH_API NormalBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward3"; }
  void release_variables() override {


  }

  std::vector<int64_t> mean_sizes;
  std::vector<int64_t> std_sizes;

};
struct TORCH_API LinalgHouseholderProductBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgHouseholderProductBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    tau_.reset_data();
  }

  SavedVariable input_;
  SavedVariable tau_;

};
struct TORCH_API OrmqrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrmqrBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API PermuteBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PermuteBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> dims;

};
struct TORCH_API PoissonBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PoissonBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API PowBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar exponent;

};
struct TORCH_API PowBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    exponent_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable exponent_;
  SavedVariable result_;

};
struct TORCH_API PowBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    result_.reset_data();
  }

  at::Scalar self;
  SavedVariable exponent_;
  SavedVariable result_;

};
struct TORCH_API ProdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API ProdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API PutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  SavedVariable index_;
  torch::autograd::generated::TypeAndSize source_info;
  bool accumulate;
  SavedVariable source_;

};
struct TORCH_API LinalgQrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgQrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    Q_.reset_data();
    R_.reset_data();
  }

  SavedVariable self_;
  std::string mode;
  SavedVariable Q_;
  SavedVariable R_;

};
struct TORCH_API Rad2DegBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rad2DegBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API RandomBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API RandomBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API RandomBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward2"; }
  void release_variables() override {


  }



};
struct TORCH_API ReciprocalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReciprocalBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API RemainderBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API RemainderBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API RenormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RenormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar p;
  int64_t dim = 0;
  at::Scalar maxnorm;

};
struct TORCH_API RepeatBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RepeatBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> repeats;

};
struct TORCH_API SpecialEntrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialEntrBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API SpecialNdtriBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialNdtriBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API ReshapeAliasBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReshapeAliasBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API RoundBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RoundBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API RsqrtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsqrtBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API ScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API ScatterBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API ScatterAddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterAddBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API SelectBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  int64_t index = 0;

};
struct TORCH_API SelectBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackwardBackward0"; }
  void release_variables() override {


  }

  int64_t dim = 0;
  int64_t index = 0;

};
struct TORCH_API SigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API LogitBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogitBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  c10::optional<double> eps;

};
struct TORCH_API SignBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SignBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API SgnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SgnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API SinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API SincBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SincBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API SinhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API SliceBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  c10::optional<int64_t> start;
  c10::optional<int64_t> end;
  int64_t step = 0;

};
struct TORCH_API SliceBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackwardBackward0"; }
  void release_variables() override {


  }

  int64_t dim = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t step = 0;

};
struct TORCH_API SlogdetBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlogdetBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    sign_.reset_data();
    logabsdet_.reset_data();
  }

  SavedVariable self_;
  SavedVariable sign_;
  SavedVariable logabsdet_;

};
struct TORCH_API LinalgSlogdetBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSlogdetBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    sign_.reset_data();
    logabsdet_.reset_data();
  }

  SavedVariable self_;
  SavedVariable sign_;
  SavedVariable logabsdet_;

};
struct TORCH_API SolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    A_.reset_data();
    solution_.reset_data();
  }

  SavedVariable self_;
  SavedVariable A_;
  SavedVariable solution_;

};
struct TORCH_API LinalgSolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    other_.reset_data();
    result_.reset_data();
  }

  SavedVariable input_;
  SavedVariable other_;
  SavedVariable result_;

};
struct TORCH_API SortBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API SortBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API SplitBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  int64_t split_size = 0;
  int64_t dim = 0;

};
struct TORCH_API UnsafeSplitBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeSplitBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  int64_t split_size = 0;
  int64_t dim = 0;

};
struct TORCH_API SplitWithSizesBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  std::vector<int64_t> split_sizes;
  int64_t dim = 0;

};
struct TORCH_API UnsafeSplitWithSizesBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeSplitWithSizesBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  std::vector<int64_t> split_sizes;
  int64_t dim = 0;

};
struct TORCH_API SqrtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqrtBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API SqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SqueezeBackward1 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct TORCH_API SqueezeBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward2"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SqueezeBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward3"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct TORCH_API StdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API StdMeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdMeanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API SubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward0"; }
  void release_variables() override {


  }

  at::ScalarType other_scalar_type;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;

};
struct TORCH_API SubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;

};
struct TORCH_API RsubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward0"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::ScalarType other_scalar_type;
  at::Scalar alpha;

};
struct TORCH_API RsubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar alpha;

};
struct TORCH_API SumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API NansumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NansumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  at::ScalarType self_scalar_type;
  SavedVariable self_;

};
struct TORCH_API NansumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NansumBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  at::ScalarType self_scalar_type;
  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API SvdHelperBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SvdHelperBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    U_.reset_data();
    S_.reset_data();
    V_.reset_data();
  }

  SavedVariable self_;
  bool some;
  bool compute_uv;
  SavedVariable U_;
  SavedVariable S_;
  SavedVariable V_;

};
struct TORCH_API SymeigBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SymeigBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    eigenvalues_.reset_data();
    eigenvectors_return_.reset_data();
  }

  SavedVariable self_;
  bool eigenvectors;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_return_;

};
struct TORCH_API LinalgEighBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgEighBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    eigenvalues_.reset_data();
    eigenvectors_.reset_data();
  }

  SavedVariable self_;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;

};
struct TORCH_API LinalgEigBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgEigBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    eigenvalues_.reset_data();
    eigenvectors_.reset_data();
  }

  SavedVariable self_;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;

};
struct TORCH_API TBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API TBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API FlipBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FlipBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> dims;

};
struct TORCH_API RollBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RollBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> shifts;
  std::vector<int64_t> dims;

};
struct TORCH_API Rot90Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rot90Backward0"; }
  void release_variables() override {


  }

  int64_t k = 0;
  std::vector<int64_t> dims;

};
struct TORCH_API TakeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TakeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  torch::autograd::generated::TypeAndSize self_info;
  SavedVariable index_;

};
struct TORCH_API TanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API TanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API TopkBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TopkBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API TraceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TraceBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API TransposeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward0"; }
  void release_variables() override {


  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TORCH_API TransposeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward1"; }
  void release_variables() override {


  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TORCH_API TriangularSolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriangularSolveBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    A_.reset_data();
    solution_.reset_data();
  }

  SavedVariable self_;
  SavedVariable A_;
  bool upper;
  bool transpose;
  bool unitriangular;
  SavedVariable solution_;

};
struct TORCH_API TrilBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilBackward0"; }
  void release_variables() override {


  }

  int64_t diagonal = 0;

};
struct TORCH_API TriuBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriuBackward0"; }
  void release_variables() override {


  }

  int64_t diagonal = 0;

};
struct TORCH_API TruncBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TruncBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ToDenseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToDenseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API ToSparseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ToSparseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API ToMkldnnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToMkldnnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API UnfoldBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dimension = 0;
  int64_t size = 0;
  int64_t step = 0;

};
struct TORCH_API UnfoldBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackwardBackward0"; }
  void release_variables() override {


  }

  int64_t dim = 0;
  int64_t size = 0;
  int64_t step = 0;

};
struct TORCH_API UniformBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniformBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueDimBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueDimBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueConsecutiveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueConsecutiveBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueDimConsecutiveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueDimConsecutiveBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API Unique2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Unique2Backward0"; }
  void release_variables() override {


  }



};
struct TORCH_API UnsafeViewBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeViewBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API UnsqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward0"; }
  void release_variables() override {


  }

  int64_t dim = 0;

};
struct TORCH_API UnsqueezeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward1"; }
  void release_variables() override {


  }

  int64_t dim = 0;

};
struct TORCH_API VarBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;

};
struct TORCH_API VarMeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarMeanBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API ViewBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API ViewAsRealBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsRealBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ViewAsComplexBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsComplexBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API SWhereBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SWhereBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    condition_.reset_data();
  }

  SavedVariable condition_;

};
struct TORCH_API WeightNormCudaInterfaceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "WeightNormCudaInterfaceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    v_.reset_data();
    g_.reset_data();
    result1_.reset_data();
  }

  SavedVariable v_;
  SavedVariable g_;
  int64_t dim = 0;
  SavedVariable result1_;

};
struct TORCH_API ZeroBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ZeroBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API SparseMaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseMaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  SavedVariable mask_;

};
struct TORCH_API SparseCooTensorWithDimsAndTensorsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseCooTensorWithDimsAndTensorsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;

};
struct TORCH_API SparseSumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;

};
struct TORCH_API StandardGammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API StandardGammaGradBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaGradBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API ValuesBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;

};
struct TORCH_API TrilinearBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilinearBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    i1_.reset_data();
    i2_.reset_data();
    i3_.reset_data();
  }

  SavedVariable i1_;
  SavedVariable i2_;
  SavedVariable i3_;
  std::vector<int64_t> expand1;
  std::vector<int64_t> expand2;
  std::vector<int64_t> expand3;
  std::vector<int64_t> sumdim;

};
struct TORCH_API ConstantPadNdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConstantPadNdBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> pad;

};
struct TORCH_API BinaryCrossEntropyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;

};
struct TORCH_API BinaryCrossEntropyBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  SavedVariable grad_output_;

};
struct TORCH_API BinaryCrossEntropyWithLogitsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyWithLogitsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    pos_weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable pos_weight_;
  int64_t reduction = 0;

};
struct TORCH_API EmbeddingBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  int64_t weight_argsize_0 = 0;
  SavedVariable indices_;
  int64_t padding_idx = 0;
  bool scale_grad_by_freq;
  bool sparse;

};
struct TORCH_API EmbeddingDenseBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingDenseBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  int64_t padding_idx = 0;

};
struct TORCH_API EmbeddingBagBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    weight_.reset_data();
    indices_.reset_data();
    offsets_.reset_data();
    per_sample_weights_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }

  SavedVariable weight_;
  SavedVariable indices_;
  SavedVariable offsets_;
  int64_t mode = 0;
  int64_t padding_idx = 0;
  int64_t weight_argsize_0 = 0;
  bool scale_grad_by_freq;
  bool sparse;
  SavedVariable per_sample_weights_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct TORCH_API EmbeddingRenormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingRenormBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API KlDivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  bool log_target;

};
struct TORCH_API L1LossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API MseLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API MultiMarginLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultiMarginLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  at::Scalar p;
  at::Scalar margin;
  SavedVariable weight_;
  int64_t reduction = 0;

};
struct TORCH_API MultilabelMarginLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultilabelMarginLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    is_target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  SavedVariable is_target_;

};
struct TORCH_API NllLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;
  SavedVariable total_weight_;

};
struct TORCH_API NllLoss2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;
  SavedVariable total_weight_;

};
struct TORCH_API SmoothL1LossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double beta;

};
struct TORCH_API HuberLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HuberLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double delta;

};
struct TORCH_API SoftMarginLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API ReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API SiluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SiluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API MishBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MishBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API EluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar alpha;
  at::Scalar scale;
  at::Scalar input_scale;

};
struct TORCH_API EluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  at::Scalar alpha;
  at::Scalar scale;
  at::Scalar input_scale;
  SavedVariable result_;

};
struct TORCH_API CeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar alpha;

};
struct TORCH_API CeluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  at::Scalar alpha;
  SavedVariable result_;

};
struct TORCH_API GeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API GluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;

};
struct TORCH_API HardshrinkBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API HardshrinkBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API HardtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar min_val;
  at::Scalar max_val;

};
struct TORCH_API LeakyReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar negative_slope;

};
struct TORCH_API LeakyReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  at::Scalar negative_slope;
  SavedVariable result_;

};
struct TORCH_API LogSigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    buffer_.reset_data();
  }

  SavedVariable self_;
  SavedVariable buffer_;

};
struct TORCH_API LogSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API SparseLogSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseLogSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API PreluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;

};
struct TORCH_API PreluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;

};
struct TORCH_API RreluWithNoiseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    noise_.reset_data();
  }

  SavedVariable self_;
  SavedVariable noise_;
  at::Scalar lower;
  at::Scalar upper;
  bool training;

};
struct TORCH_API RreluWithNoiseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    result_.reset_data();
  }

  SavedVariable noise_;
  at::Scalar lower;
  at::Scalar upper;
  bool training;
  SavedVariable result_;

};
struct TORCH_API SoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API SparseSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API SparseSparseMatmulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSparseMatmulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    other_.reset_data();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API SoftplusBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  at::Scalar beta;
  at::Scalar threshold;
  SavedVariable result_;

};
struct TORCH_API SoftshrinkBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API ThresholdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar threshold;

};
struct TORCH_API ThresholdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  at::Scalar threshold;
  SavedVariable result_;

};
struct TORCH_API ReflectionPad1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReflectionPad2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReflectionPad3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReplicationPad1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReplicationPad2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReplicationPad3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API UpsampleLinear1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleBilinear2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleBicubic2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleTrilinear3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleNearest2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleLinear1DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBilinear2DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleTrilinear3DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBicubic2DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest1DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest2DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest3DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API AdaptiveAvgPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AdaptiveAvgPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API AdaptiveMaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result1_;

};
struct TORCH_API AdaptiveMaxPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result1_;

};
struct TORCH_API AvgPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;

};
struct TORCH_API AvgPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;

};
struct TORCH_API FractionalMaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable result1_;

};
struct TORCH_API FractionalMaxPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable result1_;

};
struct TORCH_API MaxPool2DWithIndicesBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result1_;

};
struct TORCH_API MaxPool3DWithIndicesBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result1_;

};
struct TORCH_API MaxUnpool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;

};
struct TORCH_API MaxUnpool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API ConvolutionOverrideableBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionOverrideableBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int64_t groups = 0;

};
struct TORCH_API ConvolutionBackwardOverrideableBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackwardOverrideableBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  std::vector<int64_t> output_padding;
  int64_t groups = 0;

};
struct TORCH_API SlowConvTranspose2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ThnnConv2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
    finput_.reset_data();
    fgrad_input_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct TORCH_API ThnnConv2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API ConvDepthwise2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ConvDepthwise2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  int64_t self_argsize_1 = 0;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ConvDepthwise3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ConvDepthwise3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  int64_t self_argsize_1 = 0;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API SlowConv3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
    finput_.reset_data();
    fgrad_input_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct TORCH_API SlowConv3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API SlowConvDilated2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvDilated2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvDilated3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvDilated3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API Col2ImBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Col2ImBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Im2ColBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Im2ColBackward0"; }
  void release_variables() override {


  }

  int64_t self_argsize_2 = 0;
  int64_t self_argsize_3 = 0;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Im2ColBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Im2ColBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Col2ImBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Col2ImBackwardBackward0"; }
  void release_variables() override {


  }

  int64_t grad_output_argsize_2 = 0;
  int64_t grad_output_argsize_3 = 0;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API AdaptiveAvgPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
  }

  SavedVariable grad_output_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AdaptiveAvgPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
  }

  SavedVariable grad_output_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AdaptiveMaxPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AdaptiveMaxPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AvgPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AvgPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API EluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_or_result_.reset_data();
    grad_output_.reset_data();
  }

  at::Scalar alpha;
  at::Scalar scale;
  at::Scalar input_scale;
  bool is_result;
  SavedVariable self_or_result_;
  SavedVariable grad_output_;

};
struct TORCH_API FractionalMaxPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API FractionalMaxPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable grad_output_;

};
struct TORCH_API HardtanhBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar min_val;
  at::Scalar max_val;

};
struct TORCH_API KlDivBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  bool log_target;

};
struct TORCH_API L1LossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API LogSigmoidBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    buffer_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable self_;
  SavedVariable buffer_;
  SavedVariable grad_output_;

};
struct TORCH_API LogSoftmaxBackwardDataBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackwardDataBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    grad_output_.reset_data();
    self_.reset_data();
  }

  SavedVariable output_;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable self_;

};
struct TORCH_API LeakyReluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar negative_slope;

};
struct TORCH_API MaxPool2DWithIndicesBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API MaxPool3DWithIndicesBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API MaxUnpool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  SavedVariable indices_;
  std::vector<int64_t> output_size;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API MseLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API NllLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    weight_.reset_data();
  }

  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;

};
struct TORCH_API NllLoss2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    weight_.reset_data();
  }

  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;

};
struct TORCH_API RreluWithNoiseBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    noise_.reset_data();
  }

  SavedVariable self_;
  SavedVariable noise_;
  at::Scalar lower;
  at::Scalar upper;
  bool training;

};
struct TORCH_API ReflectionPad1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReflectionPad2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReflectionPad3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReplicationPad1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReplicationPad2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReplicationPad3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API SmoothL1LossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double beta;

};
struct TORCH_API HuberLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HuberLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double delta;

};
struct TORCH_API SoftplusBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    output_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable self_;
  at::Scalar beta;
  at::Scalar threshold;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TORCH_API SoftmaxBackwardDataBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackwardDataBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    self_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable output_;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable grad_output_;

};
struct TORCH_API SoftMarginLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API SoftshrinkBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API ThresholdBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  at::Scalar threshold;

};
struct TORCH_API UpsampleLinear1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleBilinear2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleBicubic2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleTrilinear3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleNearest2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleLinear1DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBilinear2DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleTrilinear3DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBicubic2DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest1DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest2DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest3DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API SigmoidBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TORCH_API TanhBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    grad_output_.reset_data();
  }

  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TORCH_API CudnnCtcLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnCtcLossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result1_.reset_data();
  }

  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API CudnnConvolutionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnConvolutionTransposeBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnConvolutionBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnGridSamplerBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnGridSamplerBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grid_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grid_;

};
struct TORCH_API CudnnAffineGridGeneratorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnAffineGridGeneratorBackward0"; }
  void release_variables() override {


  }

  int64_t N = 0;
  int64_t C = 0;
  int64_t H = 0;
  int64_t W = 0;

};
struct TORCH_API CudnnBatchNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct TORCH_API CudnnBatchNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
    reserveSpace_.reset_data();
  }

  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;
  SavedVariable reserveSpace_;

};
struct TORCH_API NnpackSpatialConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NnpackSpatialConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  SavedVariable input_;
  int64_t weight_argsize_2 = 0;
  int64_t weight_argsize_3 = 0;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API CudnnRnnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.clear();
    weight_released_ = true;
    hx_.reset_data();
    cx_.reset_data();
    dropout_state_.reset_data();
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable hx_;
  SavedVariable cx_;
  int64_t mode = 0;
  int64_t hidden_size = 0;
  int64_t proj_size = 0;
  int64_t num_layers = 0;
  bool batch_first;
  double dropout;
  bool train;
  bool bidirectional;
  std::vector<int64_t> batch_sizes;
  SavedVariable dropout_state_;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct TORCH_API CudnnRnnBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackwardBackward0"; }
  void release_variables() override {


  }


  size_t weight_size_;
};
struct TORCH_API MiopenConvolutionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenConvolutionTransposeBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenConvolutionBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenDepthwiseConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenDepthwiseConvolutionBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenBatchNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API MiopenBatchNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
  }

  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;

};
struct TORCH_API MiopenRnnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenRnnBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.clear();
    weight_released_ = true;
    hx_.reset_data();
    cx_.reset_data();
    dropout_state_.reset_data();
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable hx_;
  SavedVariable cx_;
  int64_t mode = 0;
  int64_t hidden_size = 0;
  int64_t num_layers = 0;
  bool batch_first;
  double dropout;
  bool train;
  bool bidirectional;
  std::vector<int64_t> batch_sizes;
  SavedVariable dropout_state_;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct TORCH_API MkldnnConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct TORCH_API MkldnnConvolutionBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackwardBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct TORCH_API MkldnnLinearBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnLinearBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable weight_;

};
struct TORCH_API MkldnnMaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnMaxPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result_;

};
struct TORCH_API MkldnnMaxPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnMaxPool3DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result_;

};
struct TORCH_API MkldnnAdaptiveAvgPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnAdaptiveAvgPool2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API MkldnnReshapeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnReshapeBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API FftR2CBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftR2CBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  int64_t normalization = 0;
  bool onesided;

};
struct TORCH_API FftC2RBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftC2RBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> dim;
  int64_t normalization = 0;

};
struct TORCH_API FftC2CBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftC2CBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> dim;
  int64_t normalization = 0;
  bool forward;

};
struct TORCH_API UnbindBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackward0"; }
  void release_variables() override {


  }

  int64_t dim = 0;

};
struct TORCH_API StackBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StackBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensors_.clear();
    tensors_released_ = true;
  }

  std::vector<SavedVariable> tensors_;
  bool tensors_released_ = false;
  int64_t dim = 0;
  size_t tensors_size_;
};
struct TORCH_API ThnnFusedLstmCellBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedLstmCellBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_gates_.reset_data();
    hidden_gates_.reset_data();
    cx_.reset_data();
    input_bias_.reset_data();
    hidden_bias_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  SavedVariable input_gates_;
  SavedVariable hidden_gates_;
  SavedVariable cx_;
  SavedVariable input_bias_;
  SavedVariable hidden_bias_;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API ThnnFusedGruCellBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedGruCellBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_gates_.reset_data();
    hidden_gates_.reset_data();
    hx_.reset_data();
    input_bias_.reset_data();
    hidden_bias_.reset_data();
    result1_.reset_data();
  }

  SavedVariable input_gates_;
  SavedVariable hidden_gates_;
  SavedVariable hx_;
  SavedVariable input_bias_;
  SavedVariable hidden_bias_;
  SavedVariable result1_;

};
struct TORCH_API PackPaddedSequenceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PackPaddedSequenceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  std::vector<int64_t> input_sizes;
  bool batch_first;
  SavedVariable result1_;

};
struct TORCH_API SegmentReduceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SegmentReduceBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.reset_data();
    lengths_.reset_data();
    result_.reset_data();
  }

  SavedVariable data_;
  std::string reduce;
  SavedVariable lengths_;
  SavedVariable result_;

};
struct TORCH_API PinMemoryBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PinMemoryBackward0"; }
  void release_variables() override {


  }



};

}}} // namespace torch::autograd::generated

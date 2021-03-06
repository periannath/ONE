/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnkit/support/tflite/TensorContext.h"
#include "nnkit/support/tflite/TensorUtils.h"

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Overlay.h>

namespace nnkit
{
namespace support
{
namespace tflite
{

nncc::core::ADT::tensor::Shape TensorContext::shape(uint32_t n) const
{
  return tensor_shape(_tensors.at(n));
}

void TensorContext::getMutableFloatTensor(uint32_t n,
                                          const nnkit::TensorContext::TypedAccessor<float> &f)
{
  using nncc::core::ADT::tensor::LexicalLayout;
  using nncc::core::ADT::tensor::make_overlay;

  auto t = _tensors.at(n);

  float *data = reinterpret_cast<float *>(t->data.f);
  auto overlay = make_overlay<float, LexicalLayout>(shape(n), data);

  f(*this, n, overlay);
}

void TensorContext::getConstFloatTensor(uint32_t n,
                                        const nnkit::TensorContext::TypedReader<float> &f) const
{
  using nncc::core::ADT::tensor::LexicalLayout;
  using nncc::core::ADT::tensor::make_overlay;

  auto t = _tensors.at(n);

  float *data = reinterpret_cast<float *>(t->data.f);
  auto overlay = make_overlay<float, LexicalLayout>(shape(n), data);

  f(*this, n, overlay);
}

} // namespace tflite
} // namespace support
} // namespace nnkit

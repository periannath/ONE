/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_LOGISTIC_H__
#define __NNFW_CKER_LOGISTIC_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"
#include "cker/neon/NEMath.h"

#include <cmath>
#include <Eigen/Core>

namespace nnfw
{
namespace cker
{

inline void Logistic(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                     float *output_data)
{
  // auto input_map = MapAsVector(input_data, input_shape);
  // auto output_map = MapAsVector(output_data, output_shape);
  // output_map.array() = input_map.array().unaryExpr(Eigen::internal::scalar_logistic_op<float>());

  UNUSED_RELEASE(output_shape);

  // int total_input_size = input_shape.FlatSize();
  size_t elements = input_shape.FlatSize();
  const float *in = input_data;
  float *out = output_data;
  int nn = elements >> 2;
  int remain = elements - (nn << 2);
  float32x4_t _one = vdupq_n_f32(1.f);

  for (; nn > 0; nn--)
  {
    float32x4_t _p = vld1q_f32(in);
    _p = vnegq_f32(_p);
    _p = arm_compute::vexpq_f32(_p);
    _p = vaddq_f32(_p, _one);
    float32x4_t _outp = arm_compute::vinvq_f32(_p);
    vst1q_f32(out, _outp);
    in += 4;
    out += 4;
  }
  for (; remain > 0; remain--)
  {
    *out = 1.f / (1.f + std::exp(-*in));
    in++;
    out++;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_LOGISTIC_H__

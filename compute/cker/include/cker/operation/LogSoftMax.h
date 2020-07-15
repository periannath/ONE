/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_LOGSOFTMAX_H__
#define __NNFW_CKER_LOGSOFTMAX_H__

#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/Types.h"
#include "cker/eigen/Utils.h"

#include <Eigen/Core>
#include <fixedpoint/fixedpoint.h>
#include <cmath>

namespace nnfw
{
namespace cker
{

inline void LogSoftmax(const SoftmaxParams &params, const Shape &input_shape,
                       const float *input_data, const Shape &output_shape, float *output_data)
{
  const int rank = input_shape.DimensionsCount();
  const int axis = (params.axis < 0) ? params.axis + rank : params.axis;
  const double beta = params.beta;
  const int depth = MatchingDim(input_shape, axis, output_shape, axis);

  int outer_size = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_size *= input_shape.Dims(i);
  }

  int inner_size = 1;
  for (int i = axis + 1; i < rank; ++i)
  {
    inner_size *= input_shape.Dims(i);
  }

  for (int i = 0; i < outer_size; ++i)
  {
    for (int j = 0; j < inner_size; ++j)
    {
      float max = std::numeric_limits<float>::lowest();
      for (int c = 0; c < depth; ++c)
      {
        max = std::max(max, input_data[(i * depth + c) * inner_size]);
      }

      float sum = 0.f;
      for (int c = 0; c < depth; ++c)
      {
        sum += std::exp((input_data[(i * depth + c) * inner_size + j] - max) * beta);
      }

      const float log_sum = std::log(sum);
      for (int c = 0; c < depth; ++c)
      {
        output_data[(i * depth + c) * inner_size + j] =
            (input_data[(i * depth + c) * inner_size + j] - max) * beta - log_sum;
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_LOGSOFTMAX_H__

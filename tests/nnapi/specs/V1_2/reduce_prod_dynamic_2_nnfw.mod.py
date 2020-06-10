#
# Copyright (C) 2018 The Android Open Source Project
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# refer to tanh_v1_dynamic.mod.py about the structore

# This adds reshape as the first op in a model and
# returns output of reshape, which is dynamic tensor

'''
Testing ReduceProd op when the input is dynamic.
    input[1, 3, 4, 1] shape [4]  (value of shape will be [1, 3, 4, 1])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          |        axis = 1    keepDims = True
          |             |             |
          +-------------+-------------+
          |
          |
          | dynamic tensor at compilation time but the shape will be [1, 3, 4, 1] at execution time
          |
      ReduceProd
          |
        output (dynamic tensor, [1, 1, 4, 1] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [1, 3, 4, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")
test_node_input = dynamic_layer.getTestNodeInput()

# write REDUCE_PROD test. input is `test_input`

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{1, 1, 4, 1}")

axis = Int32Scalar("axis", 1)
keepDims = True

model.Operation("REDUCE_PROD", test_node_input, axis, keepDims).To(model_output)

model_input_data = [6.4, 7.3, 19.3, -2.3,
           8.3, 2.0, 11.8, -3.4,
           22.8, 3.0, -28.7, 4.9]

model_output_data = [1211.136,	43.8,	-6536.138,	38.318]

Example({
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})

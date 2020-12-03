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

#ifndef __ONERT_BACKEND_ACL_NEON_BACKEND_CONTEXT_H__
#define __ONERT_BACKEND_ACL_NEON_BACKEND_CONTEXT_H__

#include <backend/BackendContext.h>

namespace onert
{
namespace backend
{
namespace acl_neon
{

class BackendContext : public onert::backend::BackendContext
{
public:
  BackendContext(const Backend *backend, const ir::Graph *graph,
                 std::shared_ptr<ITensorRegistry> tensor_registry = nullptr,
                 std::shared_ptr<ITensorBuilder> tensor_builder = nullptr,
                 std::shared_ptr<IConstantInitializer> constant_initializer = nullptr,
                 std::shared_ptr<IKernelGenerator> kernel_gen = nullptr,
                 std::shared_ptr<IOptimizer> optimizer = nullptr)
      : onert::backend::BackendContext(backend, graph, tensor_registry, tensor_builder,
                                       constant_initializer, kernel_gen, optimizer)
  {
  }

  ITensorRegistry *genTensors(const std::vector<onert::ir::OpSequenceIndex> &order,
                              const ir::OpSequences &op_seqs,
                              const ir::LowerInfoMap &lower_info) override;
  FunctionMap genKernels(const std::vector<onert::ir::OpSequenceIndex> &order,
                         const ir::OpSequences &op_seqs) override;

private:
  void planTensors(const std::vector<onert::ir::OpSequenceIndex> &order,
                   const ir::OpSequences &op_seqs, const ir::LowerInfoMap &lower_info);
};

} // namespace acl_neon
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_NEON_BACKEND_CONTEXT_H__
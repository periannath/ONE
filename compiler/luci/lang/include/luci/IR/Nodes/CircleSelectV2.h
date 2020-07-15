/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_CIRCLE_SELECT_V2_H__
#define __LUCI_IR_CIRCLE_SELECT_V2_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief SELECT_V2 in Circle
 */
class CircleSelectV2 final : public FixedArityNode<3, CircleNodeImpl<CircleOpcode::SELECT_V2>>
{
public:
  CircleSelectV2() = default;

public:
  loco::Node *condition(void) const { return at(0)->node(); }
  void condition(loco::Node *node) { at(0)->node(node); }

  loco::Node *t(void) const { return at(1)->node(); }
  void t(loco::Node *node) { at(1)->node(node); }

  loco::Node *e(void) const { return at(2)->node(); }
  void e(loco::Node *node) { at(2)->node(node); }
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_SELECT_V2_H__
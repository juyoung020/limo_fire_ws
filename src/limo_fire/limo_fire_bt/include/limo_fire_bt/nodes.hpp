// 트리 XML(behavior_trees/limo_fire_bt.xml) 의 노드를 공장에 등록한다
#pragma once

#include <memory>

#include <behaviortree_cpp/bt_factory.h>

#include "limo_fire_bt/ros_context.hpp"

namespace limo_fire_bt
{

void register_condition_nodes(BT::BehaviorTreeFactory & factory, const std::shared_ptr<RosContext> & ctx);
void register_motion_nodes(BT::BehaviorTreeFactory & factory, const std::shared_ptr<RosContext> & ctx);
void register_mission_nodes(BT::BehaviorTreeFactory & factory, const std::shared_ptr<RosContext> & ctx);
void register_hardware_nodes(BT::BehaviorTreeFactory & factory, const std::shared_ptr<RosContext> & ctx);

inline void register_all_nodes(BT::BehaviorTreeFactory & factory, const std::shared_ptr<RosContext> & ctx)
{
  register_condition_nodes(factory, ctx);
  register_motion_nodes(factory, ctx);
  register_mission_nodes(factory, ctx);
  register_hardware_nodes(factory, ctx);
}

// 포트 값 꺼내기: 없으면 BT::RuntimeError (트리 XML 잘못 = 개발자 실수라 바로 드러나게)
template<class T>
T need(const BT::TreeNode & node, const std::string & port)
{
  auto v = node.getInput<T>(port);
  if (!v) {throw BT::RuntimeError(node.registrationName(), ": 포트 ", port, " — ", v.error());}
  return v.value();
}

}  // namespace limo_fire_bt

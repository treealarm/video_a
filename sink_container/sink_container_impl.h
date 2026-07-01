#pragma once

#include <memory>
#include <vector>
#include <mutex>

#include "interfaces/sink_container.h"

class sink_container_impl : public sink_container
{
public:
  ~sink_container_impl() override {};
  void add_sink(std::shared_ptr<media_sink> sink) override;
  void remove_sink(std::shared_ptr<media_sink> sink) override;

  std::vector<std::shared_ptr<media_sink>> snapshot_sinks();
private:
  std::vector<std::weak_ptr<media_sink>> m_sinks;
  mutable std::mutex m_sink_mutex;
};

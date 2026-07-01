#pragma once

#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// Plain spdlog console logger — video_a is a standalone, dependency-light module and does not
// feed into vms_rec's VmsLogger JSON-envelope aggregation.
inline std::shared_ptr<spdlog::logger>& log()
{
  static std::shared_ptr<spdlog::logger> instance = spdlog::stdout_color_mt("analytics-worker");
  return instance;
}

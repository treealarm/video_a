#pragma once

#include <chrono>
#include <string>

#include "sink_container.h"

extern "C" {
#include <libavformat/avformat.h>
}

class reader : public sink_container {
public:
  virtual ~reader() = default;

  virtual bool open(const std::string& source) = 0;
  virtual AVStream* video_stream() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void pause() = 0;
  virtual void resume() = 0;
  virtual bool is_running() const = 0;
  virtual void step(int32_t steps) = 0;
  virtual void set_speed(double speed) = 0;
};

enum class player_mode {
  stopped,
  playing,
  paused,
  undefined
};

struct reader_progress
{
  std::chrono::milliseconds duration = std::chrono::milliseconds(0);
  std::chrono::milliseconds position = std::chrono::milliseconds(0);
};

#pragma once
#include <memory>
#include "media_packet.h"

class media_sink {
public:
  virtual ~media_sink() = default;
  virtual void on_packet(const std::shared_ptr<media_packet>& pkt) = 0;
};

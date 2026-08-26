#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

#include "interfaces/media_packet.h"
#include "interfaces/media_sink.h"
#include "interfaces/reader.h"

#include "sink_container/sink_container_impl.h"

extern "C" {
#include <libavformat/avformat.h>
}

// Reads a video file instead of an RTSP stream, so a watch can run against footage on disk.
//
// Two consumers with opposite needs, which is why pacing is a constructor argument rather than a
// property of the reader. A watch wants `realtime`: frame_sampler measures keyframe spacing and
// runs its emit schedule against the wall clock, so a file read at disk speed would look to it
// like a stream keying thousands of times a second and every rate decision would be nonsense. A
// batch pass wants the opposite -- there is nobody watching, and reading as fast as the sink
// accepts is the whole point.
//
// Sinks are called inline on the read loop, as they are for rtsp_reader. For a file that is a
// feature rather than a hazard: a slow sink simply slows the read, where an RTSP source would
// have back-pressured the socket. It is what lets the ROI pass decode and encode every frame
// without a queue that sheds under load.
class file_reader final : public reader {
public:
  /// `on_end` is called from the read loop when the input is exhausted, once per pass. With
  /// `loop` set the reader starts over instead and never reports an end.
  file_reader(std::string watch_id, bool loop, bool realtime,
    std::function<void()> on_end = {});
  ~file_reader() override;

  bool open(const std::string& path) override;
  void add_sink(std::shared_ptr<media_sink> sink) override;
  void remove_sink(std::shared_ptr<media_sink> sink) override;

  void start() override;
  void stop() override;

  void pause() override {};
  void resume() override {};
  bool is_running() const override
  {
    return m_running;
  }
  void step(int32_t /*steps*/) override {};
  void set_speed(double /*speed*/) override {};
  AVStream* video_stream() const override;

private:
  bool connect();
  void disconnect();
  void read_loop();
  void pace(const media_packet& pkt);

  std::string m_path;
  bool m_loop = false;
  bool m_realtime = false;
  std::function<void()> m_on_end;

  avformat_context_ptr m_ctx = nullptr;
  int m_video_index = -1;
  std::unordered_set<int> m_forward_stream_indexes;

  sink_container_impl m_sinks_helper;

  std::thread m_thread;
  std::atomic_bool m_running{ false };

  // Wall-clock anchor for realtime pacing: when the first packet of a pass was handed on, and
  // what its timestamp was. Everything after it is delayed to keep the same distance.
  std::chrono::steady_clock::time_point m_pace_origin{};
  int64_t m_pace_first_pts_us = AV_NOPTS_VALUE;

  std::string m_watch_id;
};

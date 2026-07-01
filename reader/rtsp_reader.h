#pragma once

#include <atomic>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <thread>

#include "interfaces/media_packet.h"
#include "interfaces/media_sink.h"
#include "interfaces/reader.h"

#include "sink_container/sink_container_impl.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/time.h>
}

// Forked from vms_rec's media_server/reader/rtsp_reader.* — trimmed of the ONVIF GetStreamUri
// fallback resolve path. video_a never resolves stream URLs itself: WatchRequest.rtsp_url always
// arrives already fully resolved from the caller (VmsAnalytics), so this reader has no knowledge
// of ONVIF/VmsCfg/camera models at all.
class rtsp_reader final : public reader {
public:
  rtsp_reader(const std::string& watch_id, const std::string& user, const std::string& pass);
  ~rtsp_reader() override;

  bool open(const std::string& url) override;
  void add_sink(std::shared_ptr<media_sink> sink) override;
  void remove_sink(std::shared_ptr<media_sink> sink) override;

  void start() override;
  void stop() override;

  void pause() override
  {
  };
  void resume() override
  {
  };
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

  std::string m_rtsp_url;
  avformat_context_ptr m_ctx = nullptr;
  int m_video_index = -1;
  std::unordered_set<int> m_forward_stream_indexes;

  sink_container_impl m_sinks_helper;

  std::thread m_thread;
  std::atomic_bool m_running{ false };

  std::string m_watch_id;
  std::string m_user;
  std::string m_password;
};

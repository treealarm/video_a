#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference/detection.h"

class rtsp_reader;
class pipeline;
class frame_sampler;
class crop_writer;

struct watch_params {
  std::string watch_id;
  std::string rtsp_url;
  std::string cred_user;
  std::string cred_pass;
  std::vector<detection_kind> classes;
  float min_confidence = 0.5f;
  // Informational only for Phase 1 — the effective sample rate is bound by the stream's
  // keyframe interval (see frame_sampler), not actively throttled to this value.
  uint32_t sample_fps = 1;
};

// Owns the map watch_id -> {rtsp_reader, frame_sampler, pipeline}, thread-safe. Driven entirely
// by explicit StartWatch/StopWatch calls (see project plan A.3) — video_a never decides on its
// own what to watch or pulls its own configuration.
class watch_manager {
public:
  using detection_callback = std::function<void(const std::string& watch_id, const final_detection&)>;

  watch_manager(std::string model_dir, std::shared_ptr<crop_writer> crops, detection_callback on_detection);
  ~watch_manager();

  // Idempotent: re-calling with the same watch_id tears down and recreates the watch with the
  // new params.
  bool start_watch(const watch_params& params);
  void stop_watch(const std::string& watch_id);

private:
  struct watch_entry {
    std::shared_ptr<rtsp_reader> reader;
    std::shared_ptr<frame_sampler> sampler;
    std::unique_ptr<pipeline> pipeline_instance;
  };

  void stop_watch_locked(const std::string& watch_id);

  std::string m_model_dir;
  std::shared_ptr<crop_writer> m_crops;
  detection_callback m_on_detection;

  std::mutex m_mutex;
  std::unordered_map<std::string, watch_entry> m_watches;
};

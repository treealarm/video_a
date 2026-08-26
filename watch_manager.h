#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference/detection.h"

class reader;
class pipeline;
class frame_sampler;
class inference_worker;

struct watch_params {
  std::string watch_id;
  std::string rtsp_url;
  /// Read this file rather than a stream. Mutually exclusive with rtsp_url — when set it is what
  /// the watch opens, and the credentials and listen mode below do not apply.
  std::string file_path;
  std::string cred_user;
  std::string cred_pass;
  std::vector<detection_kind> classes;
  float min_confidence = 0.5f;
  // The target rate, and an upper bound that is enforced: frame_sampler throttles to it, and
  // decodes every packet rather than keyframes alone when the stream does not key often enough to
  // supply it on the cheap path.
  uint32_t sample_fps = 1;
  bool attach_debug_crops = false;
  /// Wait to be published to rather than connecting out. `rtsp_url` is then the endpoint this
  /// worker opens, not one it dials -- see rtsp_reader's listen mode.
  bool listen_for_push = false;
};

/// What to bind, and what to tell the caller — deliberately two different strings.
///
/// Binding takes 0.0.0.0: the worker accepts the publish on whatever interface it arrives by.
/// Advertising takes ANALYTICS_ADVERTISE_HOST, because a container cannot work out the name it is
/// reachable by from outside, and only the worker can answer that question at all -- which camera
/// belongs to which worker is decided by the caller, but the address is the worker's own fact.
std::string analytics_bind_url(const std::string& watch_id);
std::string analytics_advertised_url(const std::string& watch_id);

// Owns the map watch_id -> {rtsp_reader, frame_sampler, pipeline}, thread-safe. Driven entirely
// by explicit StartWatch/StopWatch calls (see project plan A.3) — video_a never decides on its
// own what to watch or pulls its own configuration.
class watch_manager {
public:
  using detection_callback = std::function<void(const std::string& watch_id, const final_detection&)>;

  watch_manager(std::string model_dir, int reid_embed_interval_sec, detection_callback on_detection);
  ~watch_manager();

  // Idempotent: re-calling with the same watch_id tears down and recreates the watch with the
  // new params.
  bool start_watch(const watch_params& params);
  void stop_watch(const std::string& watch_id);

private:
  // Member order is load-bearing: destruction runs in reverse declaration order, and each stage's
  // thread must be joined before the object it calls into is destroyed. Reverse order here is
  // reader (stage 1) -> sampler (stage 2, joins its decode thread) -> inference (stage 3, joins its
  // detect thread) -> pipeline_instance (used by the detect thread, so destroyed last).
  struct watch_entry {
    std::unique_ptr<pipeline> pipeline_instance;
    std::shared_ptr<inference_worker> inference;
    std::shared_ptr<frame_sampler> sampler;
    std::shared_ptr<::reader> source;
  };

  void stop_watch_locked(const std::string& watch_id);

  std::string m_model_dir;
  int m_reid_embed_interval_sec;
  detection_callback m_on_detection;

  std::mutex m_mutex;
  std::unordered_map<std::string, watch_entry> m_watches;
};

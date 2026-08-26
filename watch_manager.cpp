#include "watch_manager.h"

#include <cstdlib>
#include <unistd.h>

#include "reader/rtsp_reader.h"
#include "inference/frame_sampler.h"
#include "inference/inference_worker.h"
#include "inference/pipeline.h"
#include "logging.h"

watch_manager::watch_manager(std::string model_dir, int reid_embed_interval_sec, detection_callback on_detection)
  : m_model_dir(std::move(model_dir))
  , m_reid_embed_interval_sec(reid_embed_interval_sec)
  , m_on_detection(std::move(on_detection))
{
}

watch_manager::~watch_manager()
{
  std::scoped_lock lock(m_mutex);
  for (auto& [id, entry] : m_watches)
  {
    if (entry.reader) entry.reader->stop();
  }
  m_watches.clear();
}

namespace {
const char* env_or(const char* name, const char* fallback)
{
  const char* raw = std::getenv(name);
  return (raw && *raw) ? raw : fallback;
}

/// What this machine calls itself, as the address of last resort.
///
/// It is the right answer in both places this runs, which a fixed string is not. Under compose
/// the container's hostname is the service name its peers resolve; on a developer's machine,
/// where the producer runs beside the worker rather than in the network next to it, it is the
/// machine's own name. ANALYTICS_ADVERTISE_HOST overrides it wherever neither holds -- behind a
/// NAT, say, or when the producer knows the worker by another name.
std::string own_hostname()
{
  char buf[256] = { 0 };
  if (gethostname(buf, sizeof(buf) - 1) != 0 || buf[0] == '\0')
    return "localhost";
  return buf;
}
}  // namespace

std::string analytics_bind_url(const std::string& watch_id)
{
  return std::string("rtsp://0.0.0.0:") + env_or("ANALYTICS_RTSP_PORT", "8555") + "/" + watch_id;
}

std::string analytics_advertised_url(const std::string& watch_id)
{
  const std::string host = env_or("ANALYTICS_ADVERTISE_HOST", "");
  return "rtsp://" + (host.empty() ? own_hostname() : host) + ":"
    + env_or("ANALYTICS_RTSP_PORT", "8555") + "/" + watch_id;
}

bool watch_manager::start_watch(const watch_params& params)
{
  std::scoped_lock lock(m_mutex);

  stop_watch_locked(params.watch_id); // upsert: tear down any existing watch before recreating

  pipeline_config cfg;
  cfg.watch_id = params.watch_id;
  cfg.classes = params.classes;
  cfg.min_confidence = params.min_confidence;
  cfg.attach_debug_crops = params.attach_debug_crops;
  cfg.reid_embed_interval_sec = m_reid_embed_interval_sec;

  watch_entry entry;
  entry.pipeline_instance = std::make_unique<pipeline>(cfg, m_model_dir);

  auto* pipeline_ptr = entry.pipeline_instance.get();
  auto on_detection = m_on_detection;
  const auto watch_id = params.watch_id;

  // Stage 3 (detect thread): run inference on decoded frames.
  entry.inference = std::make_shared<inference_worker>([pipeline_ptr, on_detection, watch_id](const decoded_frame& frame)
  {
    pipeline_ptr->process_frame(frame, [&](const final_detection& det) { on_detection(watch_id, det); });
  });

  // Stage 2 (decode thread): frame_sampler decodes and hands frames to the detect stage at the
  // requested rate -- keyframes only where the stream keys often enough to supply it, every packet
  // where it does not (see frame_sampler).
  auto* inference_ptr = entry.inference.get();
  entry.sampler = std::make_shared<frame_sampler>([inference_ptr](const decoded_frame& frame)
  {
    inference_ptr->submit(frame);
  }, params.sample_fps);

  entry.reader = std::make_shared<rtsp_reader>(
    params.watch_id, params.cred_user, params.cred_pass, params.listen_for_push);
  if (!entry.reader->open(params.rtsp_url))
  {
    log()->error("watch_manager: failed to open watch '{}'", params.watch_id);
    return false;
  }

  entry.reader->add_sink(entry.sampler);
  entry.reader->start();

  m_watches[params.watch_id] = std::move(entry);
  log()->info("watch_manager: started watch '{}' -> {}", params.watch_id, params.rtsp_url);
  return true;
}

void watch_manager::stop_watch(const std::string& watch_id)
{
  std::scoped_lock lock(m_mutex);
  stop_watch_locked(watch_id);
}

void watch_manager::stop_watch_locked(const std::string& watch_id)
{
  auto it = m_watches.find(watch_id);
  if (it == m_watches.end()) return;

  if (it->second.reader)
    it->second.reader->stop();

  m_watches.erase(it);
  log()->info("watch_manager: stopped watch '{}'", watch_id);
}

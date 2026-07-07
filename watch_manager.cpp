#include "watch_manager.h"

#include "reader/rtsp_reader.h"
#include "inference/frame_sampler.h"
#include "inference/inference_worker.h"
#include "inference/pipeline.h"
#include "logging.h"

watch_manager::watch_manager(std::string model_dir, detection_callback on_detection)
  : m_model_dir(std::move(model_dir))
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

bool watch_manager::start_watch(const watch_params& params)
{
  std::scoped_lock lock(m_mutex);

  stop_watch_locked(params.watch_id); // upsert: tear down any existing watch before recreating

  pipeline_config cfg;
  cfg.watch_id = params.watch_id;
  cfg.classes = params.classes;
  cfg.min_confidence = params.min_confidence;
  cfg.attach_debug_crops = params.attach_debug_crops;

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

  // Stage 2 (decode thread): frame_sampler decodes keyframes and hands them to the detect stage.
  auto* inference_ptr = entry.inference.get();
  entry.sampler = std::make_shared<frame_sampler>([inference_ptr](const decoded_frame& frame)
  {
    inference_ptr->submit(frame);
  });

  entry.reader = std::make_shared<rtsp_reader>(params.watch_id, params.cred_user, params.cred_pass);
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

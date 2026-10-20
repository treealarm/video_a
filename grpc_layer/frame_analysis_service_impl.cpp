#include "frame_analysis_service_impl.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "inference/pipeline.h"
#include "logging.h"
#include "reader/file_reader.h"
#include "roi/file_analyzer.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

namespace fa = frame_analysis::v1;

namespace {

std::optional<detection_kind> to_detection_kind(fa::ObjectKind kind)
{
  switch (kind)
  {
    case fa::OBJECT_KIND_PERSON:        return detection_kind::person;
    case fa::OBJECT_KIND_FACE:          return detection_kind::face;
    case fa::OBJECT_KIND_VEHICLE:       return detection_kind::vehicle;
    case fa::OBJECT_KIND_LICENSE_PLATE: return detection_kind::license_plate;
    default:                            return std::nullopt;
  }
}

fa::ObjectKind to_object_kind(const std::string& kind)
{
  if (kind == "person")        return fa::OBJECT_KIND_PERSON;
  if (kind == "face")          return fa::OBJECT_KIND_FACE;
  if (kind == "vehicle")       return fa::OBJECT_KIND_VEHICLE;
  if (kind == "license_plate") return fa::OBJECT_KIND_LICENSE_PLATE;
  return fa::OBJECT_KIND_UNSPECIFIED;
}

// Holds one of the worker's pass slots for the duration of a call.
class job_slot {
public:
  job_slot(std::atomic_int& running, int limit)
    : m_running(running)
  {
    int current = m_running.load();
    while (current < limit)
    {
      if (m_running.compare_exchange_weak(current, current + 1))
      {
        m_held = true;
        return;
      }
    }
  }
  ~job_slot() { if (m_held) --m_running; }
  job_slot(const job_slot&) = delete;
  job_slot& operator=(const job_slot&) = delete;

  bool held() const { return m_held; }

private:
  std::atomic_int& m_running;
  bool m_held = false;
};

struct container_facts {
  std::string codec;
  double duration_sec = 0;
  int64_t frames = 0;
  int64_t bit_rate = 0;
};

}  // namespace

frame_analysis_service_impl::frame_analysis_service_impl(
  std::string model_dir, int reid_embed_interval_sec, int max_jobs)
  : m_model_dir(std::move(model_dir))
  , m_reid_embed_interval_sec(reid_embed_interval_sec)
  , m_max_jobs(max_jobs > 0 ? max_jobs : 1)
{
}

grpc::Status frame_analysis_service_impl::AnalyzeFile(
  grpc::ServerContext* context,
  const fa::AnalyzeFileRequest* request,
  grpc::ServerWriter<fa::AnalyzeFileEvent>* writer)
{
  if (request->file_path().empty())
    return { grpc::StatusCode::INVALID_ARGUMENT, "file_path is required" };

  job_slot slot(m_running, m_max_jobs);
  if (!slot.held())
  {
    return { grpc::StatusCode::RESOURCE_EXHAUSTED,
      "already running " + std::to_string(m_max_jobs) + " file pass(es)" };
  }

  const auto started = std::chrono::steady_clock::now();
  const bool detect = request->detect_fps() >= 0;

  // The path is checked before the models are loaded: loading is the slow part, and a bad path
  // should fail in milliseconds rather than after it.
  // The same probe gives the container facts for the header: the reader opens the file on its
  // own thread later, so its stream is not there to ask when the first frame arrives.
  container_facts facts;
  {
    AVFormatContext* probe = nullptr;
    if (avformat_open_input(&probe, request->file_path().c_str(), nullptr, nullptr) < 0)
      return { grpc::StatusCode::NOT_FOUND, "cannot open " + request->file_path() };
    if (avformat_find_stream_info(probe, nullptr) >= 0)
    {
      const int v = av_find_best_stream(probe, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
      if (v >= 0)
      {
        const AVStream* st = probe->streams[v];
        if (const AVCodecDescriptor* d = avcodec_descriptor_get(st->codecpar->codec_id))
          facts.codec = d->name;
        facts.duration_sec = st->duration > 0 ? static_cast<double>(st->duration) * av_q2d(st->time_base)
          : probe->duration > 0 ? static_cast<double>(probe->duration) / AV_TIME_BASE : 0.0;
        facts.frames = st->nb_frames;
        facts.bit_rate = st->codecpar->bit_rate > 0 ? st->codecpar->bit_rate : probe->bit_rate;
      }
    }
    avformat_close_input(&probe);
  }

  std::unique_ptr<pipeline> detector;
  if (detect)
  {
    pipeline_config pcfg;
    pcfg.watch_id = "frame-analysis";
    for (const int k : request->classes())
    {
      if (const auto kind = to_detection_kind(static_cast<fa::ObjectKind>(k)))
        pcfg.classes.push_back(*kind);
    }
    if (pcfg.classes.empty())
    {
      pcfg.classes = { detection_kind::person, detection_kind::face,
        detection_kind::vehicle, detection_kind::license_plate };
    }
    if (request->min_confidence() > 0)
      pcfg.min_confidence = request->min_confidence();
    pcfg.attach_debug_crops = false;
    pcfg.reid_embed_interval_sec = m_reid_embed_interval_sec;
    if (request->has_motion_gate_threshold())
      pcfg.motion_gate_threshold = request->motion_gate_threshold();
    detector = std::make_unique<pipeline>(pcfg, m_model_dir);
  }

  std::mutex done_mutex;
  std::condition_variable done_cv;
  bool done = false;

  auto source = std::make_shared<file_reader>("frame-analysis", /*loop=*/false, /*realtime=*/false,
    [&] {
      std::scoped_lock lock(done_mutex);
      done = true;
      done_cv.notify_all();
    });
  if (!source->open(request->file_path()))
    return { grpc::StatusCode::NOT_FOUND, "cannot open " + request->file_path() };

  bool header_sent = false;
  bool client_gone = false;
  const uint32_t max_frames = request->max_frames();
  const bool include_frames = request->include_frames();

  file_analyzer_config acfg;
  acfg.detect_fps = request->detect_fps();
  acfg.scene_stats = request->include_scene_stats();
  acfg.upright = true;

  fa::AnalyzeFileEvent event;
  auto sink = std::make_shared<file_analyzer>(acfg, detector.get(),
    [&](file_analyzer& analyzer, const analyzed_frame& af) {
      if (context->IsCancelled() || (max_frames > 0 && af.index >= max_frames))
      {
        analyzer.cancel();
        done_cv.notify_all();
        return true;
      }

      if (!header_sent)
      {
        event.Clear();
        auto* h = event.mutable_header();
        h->set_width(af.frame->width);
        h->set_height(af.frame->height);
        h->set_fps_num(af.frame_rate.num);
        h->set_fps_den(af.frame_rate.den);
        h->set_time_base_num(af.time_base.num);
        h->set_time_base_den(af.time_base.den);
        h->set_pix_fmt("yuv420p");
        h->set_codec(facts.codec);
        h->set_duration_sec(facts.duration_sec);
        h->set_frame_count_estimate(facts.frames);
        h->set_bit_rate(facts.bit_rate);
        if (facts.frames == 0 && facts.duration_sec > 0 && af.frame_rate.den > 0)
        {
          h->set_frame_count_estimate(static_cast<int64_t>(
            facts.duration_sec * af.frame_rate.num / af.frame_rate.den + 0.5));
        }
        if (!writer->Write(event))
        {
          client_gone = true;
          analyzer.cancel();
          done_cv.notify_all();
          return true;
        }
        header_sent = true;
      }

      event.Clear();
      auto* f = event.mutable_frame();
      f->set_index(af.index);
      f->set_pts(af.pts);
      f->set_t_sec(af.t);
      f->set_detected(af.detected);
      for (const auto& b : *af.boxes)
      {
        auto* o = f->add_objects();
        o->set_track_id(b.track_id);
        o->set_kind(to_object_kind(b.kind));
        o->set_confidence(b.confidence);
        auto* bb = o->mutable_bbox();
        bb->set_x(b.x);
        bb->set_y(b.y);
        bb->set_width(b.width);
        bb->set_height(b.height);
      }
      if (af.stats)
      {
        auto* s = f->mutable_stats();
        s->set_mean_luma(af.stats->mean_luma);
        s->set_spatial_info(af.stats->spatial_info);
        s->set_temporal_info(af.stats->temporal_info);
        s->set_motion_dx(af.stats->motion_dx);
        s->set_motion_dy(af.stats->motion_dy);
        s->set_motion_confidence(af.stats->motion_confidence);
      }
      if (include_frames)
      {
        const auto* yuv = analyzer.pack_yuv420p(af.frame);
        if (!yuv)
          return false;
        f->set_yuv420p(yuv->data(), yuv->size());
      }
      if (!writer->Write(event))
      {
        client_gone = true;
        analyzer.cancel();
        done_cv.notify_all();
      }
      return true;
    });

  log()->info("frame analysis: {} (detect_fps={}, frames={}, stats={})", request->file_path(),
    request->detect_fps(), include_frames, acfg.scene_stats);

  source->add_sink(sink);
  source->start();
  {
    // Woken by the end of the file or by the handler giving up; the timeout is only there so a
    // client that cancels while the reader is between frames is noticed without waiting for one.
    std::unique_lock lock(done_mutex);
    while (!done && !sink->cancelled())
    {
      done_cv.wait_for(lock, std::chrono::milliseconds(200));
      if (context->IsCancelled())
        sink->cancel();
    }
  }
  source->stop();
  sink->flush();

  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();

  if (context->IsCancelled() || client_gone)
  {
    log()->info("frame analysis: cancelled after {} frames", sink->frames());
    return { grpc::StatusCode::CANCELLED, "cancelled" };
  }
  if (sink->failed())
  {
    log()->error("frame analysis: failed after {} frames", sink->frames());
    return { grpc::StatusCode::INTERNAL, "decode failed after " + std::to_string(sink->frames()) + " frames" };
  }
  if (sink->frames() == 0)
    return { grpc::StatusCode::INVALID_ARGUMENT, "no video frames decoded" };

  event.Clear();
  auto* summary = event.mutable_summary();
  summary->set_frames(sink->frames());
  summary->set_detect_passes(sink->detections_run());
  summary->set_gated_frames(detector ? detector->gated_frames() : 0);
  summary->set_elapsed_sec(elapsed);
  writer->Write(event);

  log()->info("frame analysis: {} frames, {} detection passes in {:.1f}s", sink->frames(),
    sink->detections_run(), elapsed);
  return grpc::Status::OK;
}

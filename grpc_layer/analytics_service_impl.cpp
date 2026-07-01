#include "analytics_service_impl.h"

#include <chrono>

#include "../logging.h"

namespace {
detection_kind from_proto(analytics::DetectionKind kind)
{
  switch (kind)
  {
    case analytics::PERSON: return detection_kind::person;
    case analytics::FACE: return detection_kind::face;
    case analytics::VEHICLE: return detection_kind::vehicle;
    case analytics::LICENSE_PLATE: return detection_kind::license_plate;
    default: return detection_kind::person;
  }
}

analytics::DetectionKind to_proto(detection_kind kind)
{
  switch (kind)
  {
    case detection_kind::person: return analytics::PERSON;
    case detection_kind::face: return analytics::FACE;
    case detection_kind::vehicle: return analytics::VEHICLE;
    case detection_kind::license_plate: return analytics::LICENSE_PLATE;
  }
  return analytics::DETECTION_KIND_UNSPECIFIED;
}

google::protobuf::Timestamp to_proto_timestamp(std::chrono::system_clock::time_point tp)
{
  const auto since_epoch = tp.time_since_epoch();
  const auto secs = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
  const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch - secs);

  google::protobuf::Timestamp ts;
  ts.set_seconds(secs.count());
  ts.set_nanos(static_cast<int32_t>(nanos.count()));
  return ts;
}
}

analytics_service_impl::analytics_service_impl(std::shared_ptr<watch_manager> watches, std::shared_ptr<detection_queue> queue)
  : m_watches(std::move(watches))
  , m_queue(std::move(queue))
{
}

grpc::Status analytics_service_impl::StartWatch(
  grpc::ServerContext* context,
  const analytics::WatchRequest* request,
  analytics::WatchResponse* response)
{
  watch_params params;
  params.watch_id = request->watch_id();
  params.rtsp_url = request->rtsp_url();
  params.cred_user = request->has_cred_user() ? request->cred_user() : std::string();
  params.cred_pass = request->has_cred_pass() ? request->cred_pass() : std::string();
  params.min_confidence = request->min_confidence();
  params.sample_fps = request->sample_fps();
  for (int i = 0; i < request->classes_size(); ++i)
    params.classes.push_back(from_proto(request->classes(i)));

  log()->info("StartWatch [peer={}] watch={} rtsp={} classes={} min_confidence={:.2f} sample_fps={} has_cred={}",
    context->peer(), params.watch_id, params.rtsp_url, request->classes_size(),
    params.min_confidence, params.sample_fps, request->has_cred_user());

  const bool ok = m_watches->start_watch(params);
  response->set_success(ok);
  response->set_message(ok ? "ok" : "failed to open rtsp source");
  if (!ok)
    log()->error("StartWatch failed: watch={} rtsp={}", params.watch_id, params.rtsp_url);
  return grpc::Status::OK;
}

grpc::Status analytics_service_impl::StopWatch(
  grpc::ServerContext* context,
  const analytics::StopWatchRequest* request,
  analytics::OperationStatus* response)
{
  log()->info("StopWatch [peer={}] watch={}", context->peer(), request->watch_id());
  m_watches->stop_watch(request->watch_id());
  response->set_success(true);
  return grpc::Status::OK;
}

grpc::Status analytics_service_impl::StreamDetections(
  grpc::ServerContext* context,
  const analytics::StreamDetectionsRequest* /*request*/,
  grpc::ServerWriter<analytics::DetectionEvent>* writer)
{
  log()->info("StreamDetections: client connected [peer={}]", context->peer());
  size_t sent = 0;

  while (!context->IsCancelled())
  {
    auto item = m_queue->pop_wait(std::chrono::milliseconds(500));
    if (!item) continue;

    analytics::DetectionEvent evt;
    evt.set_watch_id(item->watch_id);
    evt.set_track_id(item->detection.track_id);
    evt.set_kind(to_proto(item->detection.kind));
    evt.set_confidence(item->detection.confidence);

    auto* bbox = evt.mutable_bbox();
    bbox->set_x(item->detection.bbox.x);
    bbox->set_y(item->detection.bbox.y);
    bbox->set_width(item->detection.bbox.width);
    bbox->set_height(item->detection.bbox.height);

    *evt.mutable_detected_at() = to_proto_timestamp(item->detection.detected_at);
    evt.set_crop_ref(item->detection.crop_ref);
    if (item->detection.recognized_text) evt.set_recognized_text(*item->detection.recognized_text);
    if (item->detection.text_confidence) evt.set_text_confidence(*item->detection.text_confidence);

    if (!writer->Write(evt))
    {
      log()->warn("StreamDetections: write failed, client disconnected [peer={}] after {} events", context->peer(), sent);
      return grpc::Status::OK;
    }
    ++sent;
  }
  log()->info("StreamDetections: client cancelled [peer={}] after {} events", context->peer(), sent);
  return grpc::Status::OK;
}

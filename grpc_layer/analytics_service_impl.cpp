#include "analytics_service_impl.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "../inference/appearance_embedder.h"
#include "../inference/detection.h"
#include "../inference/face_detector.h"
#include "../inference/frame_sampler.h"
#include "../inference/plate_detector.h"
#include "../inference/primary_detector.h"
#include "../interfaces/av_deleters.h"
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

bbox_t from_proto_bbox(const analytics::BoundingBox& box)
{
  // Clamp origin first, then shrink size so x+width / y+height stay in [0, 1]. Independent
  // clamps on each field can echo a box that is not the region actually cropped for embed.
  const float x = std::clamp(box.x(), 0.0f, 1.0f);
  const float y = std::clamp(box.y(), 0.0f, 1.0f);
  const float w = std::clamp(box.width(), 0.0f, 1.0f);
  const float h = std::clamp(box.height(), 0.0f, 1.0f);
  return bbox_t{
    .x = x,
    .y = y,
    .width = std::min(w, 1.0f - x),
    .height = std::min(h, 1.0f - y),
  };
}

constexpr int kMaxJpegDim = 4096;
constexpr int64_t kMaxJpegPixels = 4096LL * 2160;
constexpr size_t kMaxJpegBytes = 8u * 1024u * 1024u;

bool jpeg_within_limits(int width, int height)
{
  if (width <= 0 || height <= 0 || width > kMaxJpegDim || height > kMaxJpegDim)
    return false;
  return static_cast<int64_t>(width) * static_cast<int64_t>(height) <= kMaxJpegPixels;
}

// Read SOF before handing the buffer to libavcodec: a ~100 KB JPEG can declare a multi-GB
// canvas, and the decoder allocates that during receive_frame — too late to reject afterwards.
std::optional<std::pair<int, int>> jpeg_sof_size(std::string_view jpeg)
{
  if (jpeg.size() < 4)
    return std::nullopt;
  const auto* p = reinterpret_cast<const uint8_t*>(jpeg.data());
  const size_t n = jpeg.size();
  if (p[0] != 0xFF || p[1] != 0xD8)
    return std::nullopt;
  size_t i = 2;
  while (i + 1 < n)
  {
    if (p[i] != 0xFF)
    {
      ++i;
      continue;
    }
    while (i < n && p[i] == 0xFF)
      ++i;
    if (i >= n)
      break;
    const uint8_t marker = p[i++];
    if (marker == 0xD8 || marker == 0x01)
      continue;
    if (marker == 0xD9 || marker == 0xDA)
      break;
    if (marker >= 0xD0 && marker <= 0xD7)
      continue;
    if (i + 1 >= n)
      break;
    const int seglen = (static_cast<int>(p[i]) << 8) | static_cast<int>(p[i + 1]);
    if (seglen < 2 || i + static_cast<size_t>(seglen) > n)
      break;
    const bool sof = (marker >= 0xC0 && marker <= 0xCF)
      && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
    if (sof)
    {
      if (seglen < 7)
        break;
      const int height = (p[i + 3] << 8) | p[i + 4];
      const int width = (p[i + 5] << 8) | p[i + 6];
      if (width <= 0 || height <= 0)
        return std::nullopt;
      return std::pair<int, int>{width, height};
    }
    i += static_cast<size_t>(seglen);
  }
  return std::nullopt;
}

std::optional<decoded_frame> decode_jpeg(const std::string& jpeg)
{
  if (jpeg.size() > kMaxJpegBytes)
    return std::nullopt;
  const auto sof = jpeg_sof_size(jpeg);
  if (!sof || !jpeg_within_limits(sof->first, sof->second))
    return std::nullopt;

  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  if (!codec)
    return std::nullopt;
  std::unique_ptr<AVCodecContext, avcodec_context_deleter> ctx(avcodec_alloc_context3(codec));
  if (!ctx)
    return std::nullopt;
  ctx->max_pixels = kMaxJpegPixels;

  if (avcodec_open2(ctx.get(), codec, nullptr) < 0)
    return std::nullopt;

  std::unique_ptr<AVFrame, avframe_deleter> frame(av_frame_alloc());
  if (!frame)
    return std::nullopt;

  std::unique_ptr<AVPacket, avpacket_deleter> pkt(av_packet_alloc());
  if (!pkt)
    return std::nullopt;
  // av_new_packet allocates size + AV_INPUT_BUFFER_PADDING_SIZE and zeroes the tail. Pointing
  // AVPacket at std::string::data() lets the MJPEG bitreader over-read the heap on a truncated JPEG.
  if (av_new_packet(pkt.get(), static_cast<int>(jpeg.size())) < 0)
    return std::nullopt;
  std::memcpy(pkt->data, jpeg.data(), jpeg.size());

  if (avcodec_send_packet(ctx.get(), pkt.get()) < 0)
    return std::nullopt;
  if (avcodec_receive_frame(ctx.get(), frame.get()) < 0)
    return std::nullopt;
  if (!jpeg_within_limits(frame->width, frame->height))
    return std::nullopt;

  std::unique_ptr<SwsContext, void(*)(SwsContext*)> sws(
    sws_getContext(
      frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
      frame->width, frame->height, AV_PIX_FMT_BGR24,
      SWS_BILINEAR, nullptr, nullptr, nullptr),
    sws_freeContext);
  if (!sws)
    return std::nullopt;

  const int buffer_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
  if (buffer_size <= 0)
    return std::nullopt;

  decoded_frame decoded;
  decoded.width = frame->width;
  decoded.height = frame->height;
  decoded.captured_at = std::chrono::system_clock::now();
  decoded.bgr.resize(static_cast<size_t>(buffer_size));

  uint8_t* dst_data[4] = { nullptr, nullptr, nullptr, nullptr };
  int dst_linesize[4] = { 0, 0, 0, 0 };
  if (av_image_fill_arrays(dst_data, dst_linesize, decoded.bgr.data(), AV_PIX_FMT_BGR24,
    frame->width, frame->height, 1) < 0)
  {
    return std::nullopt;
  }

  sws_scale(sws.get(), frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);
  return decoded;
}

std::optional<bbox_t> detect_probe_bbox(
  primary_detector& primary,
  face_detector& face,
  plate_detector& plate,
  std::mutex& detect_mu,
  const decoded_frame& frame,
  analytics::DetectionKind kind)
{
  std::lock_guard<std::mutex> lock(detect_mu);
  switch (kind)
  {
    case analytics::PERSON:
    case analytics::VEHICLE:
    {
      auto dets = primary.infer(frame);
      std::optional<bbox_t> best;
      float best_conf = 0.0f;
      const auto wanted = kind == analytics::PERSON ? detection_kind::person : detection_kind::vehicle;
      for (const auto& det : dets)
      {
        if (det.kind != wanted) continue;
        if (det.confidence > best_conf)
        {
          best_conf = det.confidence;
          best = det.bbox;
        }
      }
      return best;
    }
    case analytics::FACE:
    {
      auto faces = face.infer(frame);
      std::optional<bbox_t> best;
      float best_conf = 0.0f;
      for (const auto& det : faces)
      {
        if (det.confidence > best_conf)
        {
          best_conf = det.confidence;
          best = det.bbox;
        }
      }
      return best;
    }
    case analytics::LICENSE_PLATE:
    {
      auto plates = plate.infer(frame);
      std::optional<bbox_t> best;
      float best_conf = 0.0f;
      for (const auto& det : plates)
      {
        if (det.confidence > best_conf)
        {
          best_conf = det.confidence;
          best = det.bbox;
        }
      }
      return best;
    }
    default:
      return std::nullopt;
  }
}
}

analytics_service_impl::analytics_service_impl(
  std::shared_ptr<watch_manager> watches, std::shared_ptr<detection_queue> queue, std::string model_dir)
  : m_watches(std::move(watches))
  , m_queue(std::move(queue))
  , m_primary(std::make_unique<primary_detector>(model_dir + "/primary_detector"))
  , m_face(std::make_unique<face_detector>(model_dir + "/face_detector"))
  , m_plate(std::make_unique<plate_detector>(model_dir + "/plate_detector"))
  , m_person_embed(std::make_unique<appearance_embedder>("person_embedder", model_dir + "/person_embedder"))
  , m_face_embed(std::make_unique<appearance_embedder>("face_embedder", model_dir + "/face_embedder"))
  , m_vehicle_embed(std::make_unique<appearance_embedder>("vehicle_embedder", model_dir + "/vehicle_embedder"))
  , m_plate_embed(std::make_unique<appearance_embedder>("plate_embedder", model_dir + "/plate_embedder"))
{
}

analytics_service_impl::~analytics_service_impl() = default;

grpc::Status analytics_service_impl::StartWatch(
  grpc::ServerContext* context,
  const analytics::WatchRequest* request,
  analytics::WatchResponse* response)
{
  watch_params params;
  params.watch_id = request->watch_id();
  params.rtsp_url = request->rtsp_url();
  params.file_path = request->has_file_path() ? request->file_path() : std::string();
  params.cred_user = request->has_cred_user() ? request->cred_user() : std::string();
  params.cred_pass = request->has_cred_pass() ? request->cred_pass() : std::string();
  params.min_confidence = request->min_confidence();
  params.sample_fps = request->sample_fps();
  params.attach_debug_crops = request->attach_debug_crops();
  params.listen_for_push = request->listen_for_push();
  // In listen mode the caller does not supply a URL: it asks where to publish, and this is the
  // answer. Overwriting whatever arrived keeps one source of truth for the address -- the worker.
  if (params.listen_for_push)
    params.rtsp_url = analytics_bind_url(params.watch_id);
  for (int i = 0; i < request->classes_size(); ++i)
    params.classes.push_back(from_proto(request->classes(i)));

  const std::string& source = params.file_path.empty() ? params.rtsp_url : params.file_path;

  log()->info("StartWatch [peer={}] watch={} source={} classes={} min_confidence={:.2f} sample_fps={} has_cred={} debug_crops={}",
    context->peer(), params.watch_id, source, request->classes_size(),
    params.min_confidence, params.sample_fps, request->has_cred_user(), params.attach_debug_crops);

  const bool ok = m_watches->start_watch(params);
  response->set_success(ok);
  response->set_message(ok ? "ok" : "failed to open source");
  if (ok && params.listen_for_push)
    response->set_listen_rtsp_url(analytics_advertised_url(params.watch_id));
  if (!ok)
    log()->error("StartWatch failed: watch={} source={}", params.watch_id, source);
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
    *evt.mutable_emitted_at() = to_proto_timestamp(item->emitted_at);
    if (item->detection.recognized_text) evt.set_recognized_text(*item->detection.recognized_text);
    if (item->detection.text_confidence) evt.set_text_confidence(*item->detection.text_confidence);
    if (!item->detection.crop_jpeg.empty())
      evt.set_crop_jpeg(item->detection.crop_jpeg.data(), item->detection.crop_jpeg.size());
    if (!item->detection.embedding.empty())
    {
      *evt.mutable_embedding() = {item->detection.embedding.begin(), item->detection.embedding.end()};
      evt.set_embedding_is_fresh(item->detection.embedding_is_fresh);
    }

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

grpc::Status analytics_service_impl::EmbedImage(
  grpc::ServerContext* context,
  const analytics::EmbedImageRequest* request,
  analytics::EmbedImageResponse* response)
{
  (void)context;
  if (request->kind() == analytics::DETECTION_KIND_UNSPECIFIED)
  {
    response->set_success(false);
    response->set_message("kind is required");
    return grpc::Status::OK;
  }
  if (request->jpeg().empty())
  {
    response->set_success(false);
    response->set_message("jpeg is empty");
    return grpc::Status::OK;
  }

  try
  {
    auto frame = decode_jpeg(request->jpeg());
    if (!frame)
    {
      response->set_success(false);
      response->set_message("jpeg decode failed");
      return grpc::Status::OK;
    }

    bbox_t probe_box{ .x = 0.0f, .y = 0.0f, .width = 1.0f, .height = 1.0f };
    if (request->has_bbox())
    {
      probe_box = from_proto_bbox(request->bbox());
    }
    else if (request->detect_first())
    {
      auto detected = detect_probe_bbox(
        *m_primary, *m_face, *m_plate, m_detect_mu, *frame, request->kind());
      if (!detected)
      {
        response->set_success(false);
        response->set_message("detector found no object for requested kind");
        return grpc::Status::OK;
      }
      probe_box = *detected;
    }

    appearance_embedder* embedder = nullptr;
    switch (request->kind())
    {
      case analytics::PERSON: embedder = m_person_embed.get(); break;
      case analytics::FACE: embedder = m_face_embed.get(); break;
      case analytics::VEHICLE: embedder = m_vehicle_embed.get(); break;
      case analytics::LICENSE_PLATE: embedder = m_plate_embed.get(); break;
      default: break;
    }
    if (embedder == nullptr || !embedder->loaded())
    {
      response->set_success(false);
      response->set_message("embedder for requested kind is unavailable");
      return grpc::Status::OK;
    }

    auto embedding = embedder->embed(*frame, probe_box);
    if (embedding.empty())
    {
      response->set_success(false);
      response->set_message("embedding failed");
      return grpc::Status::OK;
    }

    response->set_success(true);
    response->set_message("ok");
    response->mutable_embedding()->Assign(embedding.begin(), embedding.end());
    auto* bbox = response->mutable_bbox();
    bbox->set_x(probe_box.x);
    bbox->set_y(probe_box.y);
    bbox->set_width(probe_box.width);
    bbox->set_height(probe_box.height);
    return grpc::Status::OK;
  }
  catch (const std::exception& ex)
  {
    // appearance_embedder::embed catches ov::Exception so a failed ReID cannot std::terminate
    // the worker (and every active watch with it). EmbedImage must do the same: detectors and
    // decode sit on the gRPC sync-server thread with no such guard.
    log()->error("EmbedImage: {}", ex.what());
    response->set_success(false);
    response->set_message("embedding failed");
    return grpc::Status::OK;
  }
}

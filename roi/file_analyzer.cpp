#include "file_analyzer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "inference/frame_sampler.h"
#include "inference/global_motion.h"
#include "inference/pipeline.h"
#include "logging.h"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

namespace {

const char* kind_name(detection_kind kind)
{
  switch (kind)
  {
    case detection_kind::person:        return "person";
    case detection_kind::face:          return "face";
    case detection_kind::vehicle:       return "vehicle";
    case detection_kind::license_plate: return "license_plate";
  }
  return "unknown";
}

// Same normalization frame_sampler does: a YUVJ format is the deprecated spelling of a full-range
// one, and handing it to sws directly logs a deprecation warning on every single frame.
AVPixelFormat normalize_pixel_format(AVPixelFormat fmt, bool& full_range)
{
  switch (fmt)
  {
    case AV_PIX_FMT_YUVJ420P: full_range = true; return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P: full_range = true; return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P: full_range = true; return AV_PIX_FMT_YUV444P;
    case AV_PIX_FMT_YUVJ440P: full_range = true; return AV_PIX_FMT_YUV440P;
    default:                  full_range = false; return fmt;
  }
}

/// True when plane 0 of `fmt` is an 8-bit luma plane that can be read as bytes in place.
bool has_8bit_luma_plane(AVPixelFormat fmt)
{
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
  if (!desc) return false;
  if (desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_PAL))
    return false;
  return desc->nb_components >= 1 && desc->comp[0].depth == 8 && desc->comp[0].plane == 0
    && desc->comp[0].step == 1;
}

float stddev(double sum, double sum_sq, size_t n)
{
  if (n == 0) return 0.f;
  const double mean = sum / static_cast<double>(n);
  const double var = std::max(0.0, sum_sq / static_cast<double>(n) - mean * mean);
  return static_cast<float>(std::sqrt(var));
}

}  // namespace

file_analyzer::file_analyzer(file_analyzer_config config, pipeline* detector, handler on_frame)
  : m_config(config)
  , m_detector(detector)
  , m_on_frame(std::move(on_frame))
{
  if (!m_detector)
    m_config.detect_fps = -1;
}

file_analyzer::~file_analyzer() = default;

void file_analyzer::on_packet(const std::shared_ptr<media_packet>& pkt)
{
  if (m_failed || m_cancelled) return;
  if (!pkt || !pkt->packet) return;
  if (pkt->media_type != AVMEDIA_TYPE_VIDEO) return;

  if (!ensure_decoder(*pkt)) return;

  m_time_base = pkt->time_base;
  if (pkt->frame_rate.num > 0 && pkt->frame_rate.den > 0)
    m_frame_rate = pkt->frame_rate;

  if (avcodec_send_packet(m_decoder.get(), pkt->packet.get()) < 0)
    return;
  drain_decoder();
}

void file_analyzer::flush()
{
  if (m_decoder && !m_failed && !m_cancelled)
  {
    avcodec_send_packet(m_decoder.get(), nullptr);
    drain_decoder();
  }
}

bool file_analyzer::ensure_decoder(const media_packet& pkt)
{
  if (m_decoder) return true;
  if (!pkt.codec_parameters) return false;

  const AVCodec* codec = avcodec_find_decoder(pkt.codec_parameters->codec_id);
  if (!codec)
  {
    log()->error("roi job: no decoder for codec_id={}",
      static_cast<int>(pkt.codec_parameters->codec_id));
    m_failed = true;
    return false;
  }

  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  if (!ctx) { m_failed = true; return false; }

  if (avcodec_parameters_to_context(ctx, pkt.codec_parameters.get()) < 0
    || avcodec_open2(ctx, codec, nullptr) < 0)
  {
    avcodec_free_context(&ctx);
    log()->error("roi job: cannot open decoder");
    m_failed = true;
    return false;
  }

  m_decoder.reset(ctx);
  return true;
}

void file_analyzer::drain_decoder()
{
  AVFrame* frame = av_frame_alloc();
  if (!frame) { m_failed = true; return; }

  while (avcodec_receive_frame(m_decoder.get(), frame) == 0)
  {
    handle_frame(frame);
    av_frame_unref(frame);
    if (m_failed || m_cancelled) break;
  }
  av_frame_free(&frame);
}

/// The frame's position in the file's own timeline. Everything that has to be reproducible
/// across runs is measured against this rather than against the clock.
double file_analyzer::frame_time_sec(const AVFrame* frame) const
{
  const int64_t pts = frame->best_effort_timestamp != AV_NOPTS_VALUE
    ? frame->best_effort_timestamp
    : frame->pts;
  if (pts == AV_NOPTS_VALUE || m_time_base.den <= 0)
  {
    return m_frame_rate.num > 0
      ? static_cast<double>(m_frames) * m_frame_rate.den / m_frame_rate.num
      : 0.0;
  }
  return static_cast<double>(pts) * m_time_base.num / m_time_base.den;
}

bool file_analyzer::detection_due(double t) const
{
  if (m_config.detect_fps < 0) return false;          // frames only
  if (m_config.detect_fps == 0) return true;          // every frame
  if (m_detect_passes == 0) return true;              // the first frame always is
  return t + 1e-9 >= m_next_detect_at;
}

void file_analyzer::run_detection(const AVFrame* frame, double t)
{
  if (!ensure_bgr_context(frame)) return;

  decoded_frame bgr;
  bgr.width = frame->width;
  bgr.height = frame->height;
  bgr.bgr.resize(static_cast<size_t>(frame->width) * frame->height * 3);
  bgr.captured_at = std::chrono::system_clock::now();

  uint8_t* dst[4] = { bgr.bgr.data(), nullptr, nullptr, nullptr };
  int lines[4] = { frame->width * 3, 0, 0, 0 };
  sws_scale(m_bgr_ctx.get(), frame->data, frame->linesize, 0, frame->height, dst, lines);

  if (const char* dump = std::getenv("ROI_DEBUG_DUMP_BGR"))
  {
    static int n = 0;
    if (n < 2)
    {
      std::string name = std::string(dump) + "/pass" + std::to_string(n) + ".ppm";
      FILE* f = std::fopen(name.c_str(), "wb");
      std::fprintf(f, "P6\n%d %d\n255\n", bgr.width, bgr.height);
      for (size_t i = 0; i < bgr.bgr.size(); i += 3)
      { std::fputc(bgr.bgr[i+2], f); std::fputc(bgr.bgr[i+1], f); std::fputc(bgr.bgr[i], f); }
      std::fclose(f);
      log()->info("roi job: dumped {} ({}x{}) at t={:.2f}", name, bgr.width, bgr.height, t);
      ++n;
    }
  }

  m_held.clear();
  m_detector->process_frame(bgr, [this](const final_detection& det) {
    roi_box box;
    box.kind = kind_name(det.kind);
    box.x = det.bbox.x;
    box.y = det.bbox.y;
    box.width = det.bbox.width;
    box.height = det.bbox.height;
    box.track_id = det.track_id;
    box.confidence = det.confidence;
    m_held.push_back(std::move(box));
  });

  ++m_detect_passes;
  if (m_config.detect_fps > 0)
  {
    // Advanced from the deadline, not from this frame: measuring from where we actually landed
    // would let the interval drift longer by however late each frame was.
    const double period = 1.0 / m_config.detect_fps;
    m_next_detect_at = (m_detect_passes == 1 ? t : m_next_detect_at) + period;
    if (m_next_detect_at <= t)
      m_next_detect_at = t + period;
  }
}

bool file_analyzer::ensure_bgr_context(const AVFrame* frame)
{
  const auto format = static_cast<AVPixelFormat>(frame->format);
  if (m_bgr_ctx && frame->width == m_bgr_width && frame->height == m_bgr_height
    && format == m_bgr_format)
    return true;

  bool full_range = false;
  const AVPixelFormat src = normalize_pixel_format(format, full_range);
  m_bgr_ctx.reset(sws_getContext(frame->width, frame->height, src,
    frame->width, frame->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr));
  if (!m_bgr_ctx)
  {
    log()->error("roi job: cannot build BGR converter");
    m_failed = true;
    return false;
  }
  if (full_range)
  {
    const int* coeffs = sws_getCoefficients(SWS_CS_ITU601);
    sws_setColorspaceDetails(m_bgr_ctx.get(), coeffs, 1, coeffs, 1, 0, 1 << 16, 1 << 16);
  }
  m_bgr_width = frame->width;
  m_bgr_height = frame->height;
  m_bgr_format = format;
  return true;
}

/// Tightly packed YUV420P. A frame already in that format is only repacked -- its planes carry
/// alignment padding the wire does not -- and anything else is converted.
const std::vector<uint8_t>* file_analyzer::pack_yuv420p(const AVFrame* frame)
{
  const int needed = av_image_get_buffer_size(AV_PIX_FMT_YUV420P,
    frame->width, frame->height, 1);
  if (needed < 0)
  {
    m_failed = true;
    return nullptr;
  }
  m_yuv.resize(static_cast<size_t>(needed));

  const auto format = static_cast<AVPixelFormat>(frame->format);
  if (format == AV_PIX_FMT_YUV420P)
  {
    if (av_image_copy_to_buffer(m_yuv.data(), needed, frame->data, frame->linesize,
          format, frame->width, frame->height, 1) < 0)
      return nullptr;
    return &m_yuv;
  }

  bool full_range = false;
  const AVPixelFormat src = normalize_pixel_format(format, full_range);
  if (!m_yuv_ctx || frame->width != m_yuv_width || frame->height != m_yuv_height
    || format != m_yuv_format)
  {
    m_yuv_ctx.reset(sws_getContext(frame->width, frame->height, src,
      frame->width, frame->height, AV_PIX_FMT_YUV420P,
      SWS_BILINEAR, nullptr, nullptr, nullptr));
    if (!m_yuv_ctx)
    {
      log()->error("roi job: cannot build YUV420P converter");
      m_failed = true;
      return nullptr;
    }
    if (full_range)
    {
      const int* coeffs = sws_getCoefficients(SWS_CS_ITU601);
      sws_setColorspaceDetails(m_yuv_ctx.get(), coeffs, 1, coeffs, 0, 0, 1 << 16, 1 << 16);
    }
    m_yuv_width = frame->width;
    m_yuv_height = frame->height;
    m_yuv_format = format;
  }

  uint8_t* dst[4] = { nullptr, nullptr, nullptr, nullptr };
  int lines[4] = { 0, 0, 0, 0 };
  if (av_image_fill_arrays(dst, lines, m_yuv.data(), AV_PIX_FMT_YUV420P,
        frame->width, frame->height, 1) < 0)
  {
    m_failed = true;
    return nullptr;
  }
  sws_scale(m_yuv_ctx.get(), frame->data, frame->linesize, 0, frame->height, dst, lines);
  return &m_yuv;
}

frame_scene_stats file_analyzer::measure_scene(const AVFrame* frame)
{
  frame_scene_stats stats;

  const uint8_t* luma = nullptr;
  int stride = 0;
  if (has_8bit_luma_plane(static_cast<AVPixelFormat>(frame->format)))
  {
    luma = frame->data[0];
    stride = frame->linesize[0];
  }
  else if (const auto* yuv = pack_yuv420p(frame))
  {
    luma = yuv->data();
    stride = frame->width;
  }
  if (!luma || stride <= 0)
    return stats;

  const int w = frame->width;
  const int h = frame->height;

  // A grid about 320 samples wide: enough to tell texture from flat and motion from stillness,
  // independent of the source resolution.
  const int step = std::max(1, w / 320);
  const int sw = w / step;
  const int sh = h / step;
  if (sw < 3 || sh < 3)
    return stats;

  std::vector<uint8_t> sub(static_cast<size_t>(sw) * sh);
  double luma_sum = 0;
  for (int y = 0; y < sh; ++y)
  {
    const uint8_t* row = luma + static_cast<size_t>(y) * step * stride;
    uint8_t* out = sub.data() + static_cast<size_t>(y) * sw;
    for (int x = 0; x < sw; ++x)
    {
      out[x] = row[static_cast<size_t>(x) * step];
      luma_sum += out[x];
    }
  }
  stats.mean_luma = static_cast<float>(luma_sum / static_cast<double>(sub.size()));

  // Sobel magnitude over the interior.
  double si_sum = 0, si_sq = 0;
  size_t si_n = 0;
  for (int y = 1; y < sh - 1; ++y)
  {
    const uint8_t* r0 = sub.data() + static_cast<size_t>(y - 1) * sw;
    const uint8_t* r1 = r0 + sw;
    const uint8_t* r2 = r1 + sw;
    for (int x = 1; x < sw - 1; ++x)
    {
      const int gx = (r0[x + 1] + 2 * r1[x + 1] + r2[x + 1]) - (r0[x - 1] + 2 * r1[x - 1] + r2[x - 1]);
      const int gy = (r2[x - 1] + 2 * r2[x] + r2[x + 1]) - (r0[x - 1] + 2 * r0[x] + r0[x + 1]);
      const double mag = std::sqrt(static_cast<double>(gx * gx + gy * gy));
      si_sum += mag;
      si_sq += mag * mag;
      ++si_n;
    }
  }
  stats.spatial_info = stddev(si_sum, si_sq, si_n);

  if (m_prev_luma.size() == sub.size())
  {
    double ti_sum = 0, ti_sq = 0;
    for (size_t i = 0; i < sub.size(); ++i)
    {
      const double d = static_cast<double>(sub[i]) - m_prev_luma[i];
      ti_sum += d;
      ti_sq += d * d;
    }
    stats.temporal_info = stddev(ti_sum, ti_sq, sub.size());
  }
  m_prev_luma = std::move(sub);

  auto grid = luma_motion_grid(luma, stride, w, h);
  if (!m_prev_grid.empty() && !grid.empty())
  {
    if (const auto motion = estimate_global_motion(m_prev_grid, grid))
    {
      stats.motion_dx = motion->dx;
      stats.motion_dy = motion->dy;
      stats.motion_confidence = motion->confidence;
    }
  }
  m_prev_grid = std::move(grid);

  return stats;
}

void file_analyzer::handle_frame(const AVFrame* frame)
{
  if (frame->width <= 0 || frame->height <= 0) return;

  const double t = frame_time_sec(frame);
  const bool detected = detection_due(t);
  if (detected)
    run_detection(frame, t);
  if (m_failed) return;

  // The encoder rejects a timeline that goes backwards, and a file with B-frames or a broken
  // timestamp can hand us one. Nudging is right where the decoder's order is already correct
  // and only the values are not; it costs one tick of the time base.
  int64_t pts = frame->best_effort_timestamp != AV_NOPTS_VALUE
    ? frame->best_effort_timestamp
    : frame->pts;
  if (pts == AV_NOPTS_VALUE)
    pts = m_last_pts + 1;
  if (m_frames > 0 && pts <= m_last_pts)
    pts = m_last_pts + 1;

  analyzed_frame out;
  out.frame = frame;
  out.index = m_frames;
  out.pts = pts;
  out.t = t;
  out.detected = detected;
  out.boxes = &m_held;
  out.time_base = m_time_base;
  out.frame_rate = m_frame_rate;
  if (m_config.scene_stats)
    out.stats = measure_scene(frame);
  if (m_failed) return;

  if (m_on_frame && !m_on_frame(*this, out))
  {
    m_failed = true;
    return;
  }

  m_last_pts = pts;
  ++m_frames;
}

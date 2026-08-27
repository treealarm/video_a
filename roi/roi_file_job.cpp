#include "roi_file_job.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "interfaces/av_deleters.h"
#include "interfaces/media_sink.h"
#include "inference/frame_sampler.h"
#include "inference/pipeline.h"
#include "logging.h"
#include "reader/file_reader.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
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

std::string json_escape(const std::string& s)
{
  std::string out;
  out.reserve(s.size() + 8);
  for (const unsigned char c : s)
  {
    switch (c)
    {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (c < 0x20)
        {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        }
        else
        {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

struct sidecar_frame {
  double t = 0;
  std::vector<roi_box> boxes;
};

struct qpmap_sidecar_frame {
  double t = 0;
  int cols = 0;
  int rows = 0;
  std::vector<int8_t> deltas;
};

// Optional companion to the boxes sidecar: the per-block QP grid that was sent when
// --regions qp_map. Written as one document so a later offline encode can replay it.
struct qpmap_sidecar {
  int width = 0;
  int height = 0;
  int block = 0;
  std::string input;
  std::string output;
  std::vector<qpmap_sidecar_frame> frames;

  bool write(const std::string& path) const
  {
    std::ofstream out(path, std::ios::trunc);
    if (!out)
      return false;

    out << "{\n";
    out << "  \"width\": " << width << ",\n";
    out << "  \"height\": " << height << ",\n";
    out << "  \"block\": " << block << ",\n";
    out << "  \"input\": \"" << json_escape(input) << "\",\n";
    out << "  \"output\": \"" << json_escape(output) << "\",\n";
    out << "  \"frames\": [\n";
    for (size_t i = 0; i < frames.size(); ++i)
    {
      const auto& f = frames[i];
      out << "    {\"t\": " << f.t
          << ", \"cols\": " << f.cols
          << ", \"rows\": " << f.rows
          << ", \"deltas\": [";
      for (size_t j = 0; j < f.deltas.size(); ++j)
      {
        if (j) out << ",";
        out << static_cast<int>(f.deltas[j]);
      }
      out << "]}" << (i + 1 == frames.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
  }
};

// The regions of every frame, in the file's own timeline. Written as one document at the end
// rather than streamed, because a player wants the frame size in the header and that is only
// known once the first frame has been decoded.
struct sidecar {
  int width = 0;
  int height = 0;
  double fps = 0;
  std::string input;
  std::string output;
  std::vector<sidecar_frame> frames;

  bool write(const std::string& path) const
  {
    std::ofstream out(path, std::ios::trunc);
    if (!out)
      return false;

    out << "{\n";
    out << "  \"width\": " << width << ",\n";
    out << "  \"height\": " << height << ",\n";
    out << "  \"fps\": " << fps << ",\n";
    out << "  \"input\": \"" << json_escape(input) << "\",\n";
    out << "  \"output\": \"" << json_escape(output) << "\",\n";
    out << "  \"frames\": [\n";
    for (size_t i = 0; i < frames.size(); ++i)
    {
      const auto& f = frames[i];
      out << "    {\"t\": " << f.t << ", \"boxes\": [";
      for (size_t j = 0; j < f.boxes.size(); ++j)
      {
        const auto& b = f.boxes[j];
        if (j) out << ", ";
        out << "{\"kind\": \"" << json_escape(b.kind) << "\""
            << ", \"x\": " << b.x
            << ", \"y\": " << b.y
            << ", \"w\": " << b.width
            << ", \"h\": " << b.height
            << ", \"track_id\": " << b.track_id
            << ", \"confidence\": " << b.confidence << "}";
      }
      out << "]}" << (i + 1 == frames.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
  }
};

struct sws_context_deleter {
  void operator()(SwsContext* ctx) const noexcept { if (ctx) sws_freeContext(ctx); }
};
using sws_context_ptr = std::unique_ptr<SwsContext, sws_context_deleter>;
using avcodec_context_ptr = std::unique_ptr<AVCodecContext, avcodec_context_deleter>;

// Decodes every video packet the reader hands it, runs inference on a schedule of its own, and
// pushes each frame to the encoder with the regions in force at that moment.
//
// Everything here runs on the reader's thread, one packet at a time. That is what removes the
// need for a queue: the reader cannot get ahead of the encoder because it is the encoder's call
// stack that it is waiting on.
class roi_encode_sink final : public media_sink {
public:
  roi_encode_sink(const roi_job_config& config, pipeline& detector, roi_stream_client& client)
    : m_config(config)
    , m_detector(detector)
    , m_client(client)
  {
    m_sidecar.input = config.input_path;
    m_sidecar.output = config.encode.output_path;
    m_qpmap.input = config.input_path;
    m_qpmap.output = config.encode.output_path;
  }

  void on_packet(const std::shared_ptr<media_packet>& pkt) override
  {
    if (m_failed) return;
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

  /// Pushes the last frames out of the decoder. The reader has no way to say "that was the end"
  /// through media_sink, so the job calls this once the read loop has finished.
  void flush()
  {
    if (m_decoder && !m_failed)
    {
      avcodec_send_packet(m_decoder.get(), nullptr);
      drain_decoder();
    }
  }

  bool failed() const { return m_failed; }
  int64_t frames_sent() const { return m_frames_sent; }
  int64_t detections_run() const { return m_detect_passes; }
  const sidecar& regions() const { return m_sidecar; }
  const qpmap_sidecar& qp_maps() const { return m_qpmap; }

private:
  bool ensure_decoder(const media_packet& pkt)
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

  void drain_decoder()
  {
    AVFrame* frame = av_frame_alloc();
    if (!frame) { m_failed = true; return; }

    while (avcodec_receive_frame(m_decoder.get(), frame) == 0)
    {
      handle_frame(frame);
      av_frame_unref(frame);
      if (m_failed) break;
    }
    av_frame_free(&frame);
  }

  bool ensure_encoder_open(const AVFrame* frame)
  {
    if (m_opened) return true;

    m_sidecar.width = frame->width;
    m_sidecar.height = frame->height;
    m_sidecar.fps = m_frame_rate.den > 0
      ? static_cast<double>(m_frame_rate.num) / m_frame_rate.den
      : 0.0;

    std::string error;
    if (!m_client.open(m_config.encode, frame->width, frame->height,
          m_time_base.num, m_time_base.den, m_frame_rate.num, m_frame_rate.den,
          AV_PIX_FMT_YUV420P, error))
    {
      log()->error("roi job: encoder refused the stream: {}", error);
      m_failed = true;
      return false;
    }
    m_opened = true;
    log()->info("roi job: encoding {}x{} @ {:.2f} fps -> {}", frame->width, frame->height,
      m_sidecar.fps, m_config.encode.output_path);
    return true;
  }

  /// The frame's position in the file's own timeline. Everything that has to be reproducible
  /// across runs is measured against this rather than against the clock.
  double frame_time_sec(const AVFrame* frame) const
  {
    const int64_t pts = frame->best_effort_timestamp != AV_NOPTS_VALUE
      ? frame->best_effort_timestamp
      : frame->pts;
    if (pts == AV_NOPTS_VALUE || m_time_base.den <= 0)
    {
      return m_frame_rate.num > 0
        ? static_cast<double>(m_frames_sent) * m_frame_rate.den / m_frame_rate.num
        : 0.0;
    }
    return static_cast<double>(pts) * m_time_base.num / m_time_base.den;
  }

  bool detection_due(double t) const
  {
    if (m_config.detect_fps <= 0) return true;         // every frame
    if (m_detect_passes == 0) return true;             // the first frame always is
    return t + 1e-9 >= m_next_detect_at;
  }

  void run_detection(const AVFrame* frame, double t)
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

    m_held.clear();
    m_detector.process_frame(bgr, [this](const final_detection& det) {
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

  bool ensure_bgr_context(const AVFrame* frame)
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

  /// Tightly packed YUV420P, which is what the encoder was opened with. A frame already in that
  /// format is only repacked — its planes carry alignment padding the wire does not — and anything
  /// else is converted.
  bool pack_yuv420p(const AVFrame* frame)
  {
    const int needed = av_image_get_buffer_size(AV_PIX_FMT_YUV420P,
      frame->width, frame->height, 1);
    if (needed < 0)
    {
      m_failed = true;
      return false;
    }
    m_yuv.resize(static_cast<size_t>(needed));

    const auto format = static_cast<AVPixelFormat>(frame->format);
    if (format == AV_PIX_FMT_YUV420P)
    {
      return av_image_copy_to_buffer(m_yuv.data(), needed, frame->data, frame->linesize,
        format, frame->width, frame->height, 1) >= 0;
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
        return false;
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
      return false;
    }
    sws_scale(m_yuv_ctx.get(), frame->data, frame->linesize, 0, frame->height, dst, lines);
    return true;
  }

  /// The kind's signed QP delta, or nullopt when this kind has no override (skip painting).
  const int32_t* kind_qp_delta(const std::string& kind) const
  {
    for (const auto& k : m_config.encode.kinds)
      if (k.kind == kind)
        return &k.qp_delta;
    return nullptr;
  }

  double kind_pad(const std::string& kind) const
  {
    for (const auto& k : m_config.encode.kinds)
      if (k.kind == kind)
        return k.pad;
    return m_config.encode.pad;
  }

  int qpmap_block_size() const
  {
    if (m_config.qpmap_block > 0)
      return m_config.qpmap_block;
    // Match smartvideo EncoderCapabilities for libx264 / libx265.
    return m_config.encode.codec == "h264" ? 16 : 32;
  }

  /// Paints held boxes onto a coding-block QP grid. More-negative deltas win on overlap
  /// (stronger protection). Pads are applied before bodies so a body can override its halo.
  void build_qp_map(int frame_width, int frame_height)
  {
    const int block = qpmap_block_size();
    m_qp_block = block;
    m_qp_cols = std::max(1, (frame_width + block - 1) / block);
    m_qp_rows = std::max(1, (frame_height + block - 1) / block);
    m_qp_width = frame_width;
    m_qp_height = frame_height;
    const int8_t bg = static_cast<int8_t>(std::clamp(
      m_config.encode.has_background_qp_delta ? m_config.encode.background_qp_delta : 6,
      -51, 51));
    m_qp_deltas.assign(static_cast<size_t>(m_qp_cols) * m_qp_rows, bg);

    auto paint = [&](double nx0, double ny0, double nx1, double ny1, int8_t delta) {
      const int x0 = std::clamp(static_cast<int>(std::floor(nx0 * frame_width)), 0, frame_width);
      const int y0 = std::clamp(static_cast<int>(std::floor(ny0 * frame_height)), 0, frame_height);
      const int x1 = std::clamp(static_cast<int>(std::ceil(nx1 * frame_width)), 0, frame_width);
      const int y1 = std::clamp(static_cast<int>(std::ceil(ny1 * frame_height)), 0, frame_height);
      if (x1 <= x0 || y1 <= y0)
        return;
      const int c0 = x0 / block;
      const int r0 = y0 / block;
      const int c1 = (x1 - 1) / block;
      const int r1 = (y1 - 1) / block;
      for (int r = r0; r <= r1; ++r)
      {
        for (int c = c0; c <= c1; ++c)
        {
          int8_t& cell = m_qp_deltas[static_cast<size_t>(r) * m_qp_cols + c];
          if (delta < cell)
            cell = delta;
        }
      }
    };

    for (const auto& b : m_held)
    {
      const int32_t* body = kind_qp_delta(b.kind);
      if (!body)
        continue;
      const double pad = kind_pad(b.kind);
      if (pad > 0.0)
      {
        // Halo is half the body's offset toward the background -- same rule as the encoder.
        const int32_t bg_i = m_config.encode.has_background_qp_delta
          ? m_config.encode.background_qp_delta : 6;
        const int8_t pad_delta = static_cast<int8_t>(std::clamp(
          (*body + bg_i) / 2, -51, 51));
        paint(b.x - pad, b.y - pad, b.x + b.width + pad, b.y + b.height + pad, pad_delta);
      }
    }
    for (const auto& b : m_held)
    {
      const int32_t* body = kind_qp_delta(b.kind);
      if (!body)
        continue;
      paint(b.x, b.y, b.x + b.width, b.y + b.height,
        static_cast<int8_t>(std::clamp(*body, -51, 51)));
    }
  }

  /// The kind's strength as a mask value. A mask carries one scale rather than a delta per
  /// region, so the per-kind offsets are placed on it: the background sits at 0, the strongest
  /// kind asked for at 255, and everything else in between. The service reads the same
  /// `strongest` off the kind list when it is not told otherwise, so the two ends agree without
  /// either having to send the mapping.
  uint8_t mask_level(const std::string& kind) const
  {
    if (m_config.encode.kinds.empty())
      return 255;

    int32_t strongest = 0;
    const int32_t* mine = nullptr;
    for (const auto& k : m_config.encode.kinds)
    {
      strongest = std::min(strongest, k.qp_delta);
      if (k.kind == kind)
        mine = &k.qp_delta;
    }
    const int32_t background =
      m_config.encode.has_background_qp_delta ? m_config.encode.background_qp_delta : 6;
    if (!mine || strongest >= background)
      return 0;
    if (*mine >= background)
      return 0;  // this kind asks for no better than the background

    const double span = static_cast<double>(background - strongest);
    const double level = 255.0 * (background - *mine) / span;
    return static_cast<uint8_t>(std::clamp(level, 1.0, 255.0));
  }

  /// Paints the held regions into the mask, strongest value wins where they overlap.
  void build_mask(int frame_width, int frame_height)
  {
    const int scale = std::max(1, m_config.mask_scale);
    m_mask_width = std::max(1, (frame_width + scale - 1) / scale);
    m_mask_height = std::max(1, (frame_height + scale - 1) / scale);
    m_mask.assign(static_cast<size_t>(m_mask_width) * m_mask_height, 0);

    for (const auto& b : m_held)
    {
      const uint8_t level = mask_level(b.kind);
      if (level == 0)
        continue;

      // The halo the encoder adds around a box in box mode has to be added here instead: it
      // does not pad a mask, on the grounds that a mask already says what it means everywhere.
      double pad = 0;
      for (const auto& k : m_config.encode.kinds)
        if (k.kind == b.kind)
          pad = k.pad;

      // Outward rounding on every edge, because at one eighth of the frame a small object is
      // a fraction of a mask pixel and truncation would drop it entirely -- and a plate is
      // exactly the kind of small object this exists for.
      const auto lo = [](double v, int limit) {
        return std::clamp(static_cast<int>(std::floor(v)), 0, limit);
      };
      const auto hi = [](double v, int limit) {
        return std::clamp(static_cast<int>(std::ceil(v)), 0, limit);
      };
      const int x0 = lo((b.x - pad) * m_mask_width, m_mask_width);
      const int y0 = lo((b.y - pad) * m_mask_height, m_mask_height);
      const int x1 = hi((b.x + b.width + pad) * m_mask_width, m_mask_width);
      const int y1 = hi((b.y + b.height + pad) * m_mask_height, m_mask_height);

      for (int y = y0; y < y1; ++y)
      {
        uint8_t* row = m_mask.data() + static_cast<size_t>(y) * m_mask_width;
        for (int x = x0; x < x1; ++x)
          row[x] = std::max(row[x], level);
      }
    }
  }

  void handle_frame(const AVFrame* frame)
  {
    if (frame->width <= 0 || frame->height <= 0) return;
    if (!ensure_encoder_open(frame)) return;

    const double t = frame_time_sec(frame);
    if (detection_due(t))
      run_detection(frame, t);
    if (m_failed) return;

    if (!pack_yuv420p(frame)) return;

    // The encoder rejects a timeline that goes backwards, and a file with B-frames or a broken
    // timestamp can hand us one. Nudging is right where the decoder's order is already correct
    // and only the values are not; it costs one tick of the time base.
    int64_t pts = frame->best_effort_timestamp != AV_NOPTS_VALUE
      ? frame->best_effort_timestamp
      : frame->pts;
    if (pts == AV_NOPTS_VALUE)
      pts = m_last_pts + 1;
    if (m_frames_sent > 0 && pts <= m_last_pts)
      pts = m_last_pts + 1;

    bool sent = false;
    if (m_config.regions == roi_region_form::mask)
    {
      build_mask(frame->width, frame->height);
      sent = m_client.send_frame_mask(pts, m_yuv.data(), m_yuv.size(),
        m_mask.data(), m_mask_width, m_mask_height);
    }
    else if (m_config.regions == roi_region_form::qp_map)
    {
      build_qp_map(frame->width, frame->height);
      sent = m_client.send_frame_qp_map(pts, m_yuv.data(), m_yuv.size(),
        m_qp_block, m_qp_cols, m_qp_rows, m_qp_width, m_qp_height,
        m_qp_deltas.data(), m_qp_deltas.size());
      if (sent && !m_config.qpmap_json_path.empty())
      {
        if (m_qpmap.width == 0)
        {
          m_qpmap.width = frame->width;
          m_qpmap.height = frame->height;
          m_qpmap.block = m_qp_block;
        }
        m_qpmap.frames.push_back(
          qpmap_sidecar_frame{ t, m_qp_cols, m_qp_rows, m_qp_deltas });
      }
    }
    else
    {
      sent = m_client.send_frame(pts, m_yuv.data(), m_yuv.size(), m_held);
    }

    if (!sent)
    {
      log()->error("roi job: the encoder stopped accepting frames after {}", m_frames_sent);
      m_failed = true;
      return;
    }

    m_sidecar.frames.push_back(sidecar_frame{ t, m_held });
    m_last_pts = pts;
    ++m_frames_sent;
  }

  const roi_job_config& m_config;
  pipeline& m_detector;
  roi_stream_client& m_client;

  avcodec_context_ptr m_decoder;
  AVRational m_time_base{ 1, 1000 };
  AVRational m_frame_rate{ 25, 1 };

  sws_context_ptr m_bgr_ctx;
  int m_bgr_width = 0;
  int m_bgr_height = 0;
  AVPixelFormat m_bgr_format = AV_PIX_FMT_NONE;

  sws_context_ptr m_yuv_ctx;
  int m_yuv_width = 0;
  int m_yuv_height = 0;
  AVPixelFormat m_yuv_format = AV_PIX_FMT_NONE;
  std::vector<uint8_t> m_yuv;

  // The regions in force. Carried forward between detection passes, which is what makes a clip
  // detected at 2 Hz encodable at 30 fps without the boxes flickering in and out.
  std::vector<roi_box> m_held;
  double m_next_detect_at = 0;
  int64_t m_detect_passes = 0;

  std::vector<uint8_t> m_mask;
  int m_mask_width = 0;
  int m_mask_height = 0;

  std::vector<int8_t> m_qp_deltas;
  int m_qp_block = 0;
  int m_qp_cols = 0;
  int m_qp_rows = 0;
  int m_qp_width = 0;
  int m_qp_height = 0;

  bool m_opened = false;
  bool m_failed = false;
  int64_t m_frames_sent = 0;
  int64_t m_last_pts = 0;

  sidecar m_sidecar;
  qpmap_sidecar m_qpmap;
};

}  // namespace

// Resolve `codec=same` and `gop=0` against the input before the encoder opens. Resolution and
// fps already travel with every frame; these two do not, because raw frames carry no codec id
// and the service's own GOP default is "two seconds" -- fine for a live stream, wrong for an
// archive clip that has to sit next to its un-rewritten neighbours.
void match_source_encode_defaults(roi_encode_settings& settings, const std::string& path)
{
  const bool want_codec = settings.codec.empty() || settings.codec == "same";
  const bool want_gop = settings.gop <= 0;
  if (!want_codec && !want_gop)
    return;

  AVFormatContext* ctx = nullptr;
  if (avformat_open_input(&ctx, path.c_str(), nullptr, nullptr) < 0)
  {
    log()->warn("roi job: cannot probe '{}'; leaving codec/gop as asked", path);
    return;
  }
  if (avformat_find_stream_info(ctx, nullptr) < 0)
  {
    avformat_close_input(&ctx);
    log()->warn("roi job: no stream info for '{}'; leaving codec/gop as asked", path);
    return;
  }

  int video = -1;
  for (unsigned i = 0; i < ctx->nb_streams; ++i)
  {
    if (ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
    {
      video = static_cast<int>(i);
      break;
    }
  }
  if (video < 0)
  {
    avformat_close_input(&ctx);
    log()->warn("roi job: no video in '{}'; leaving codec/gop as asked", path);
    return;
  }

  AVStream* st = ctx->streams[video];
  if (want_codec)
  {
    switch (st->codecpar->codec_id)
    {
      case AV_CODEC_ID_HEVC: settings.codec = "h265"; break;
      case AV_CODEC_ID_H264: settings.codec = "h264"; break;
      default:
        // Same fallback the service uses for SameAsSource on an unrecognised input.
        settings.codec = "h264";
        log()->warn("roi job: source codec_id={} is neither h264 nor h265; encoding as h264",
          static_cast<int>(st->codecpar->codec_id));
        break;
    }
  }

  if (want_gop)
  {
    // First closed gap between keyframes, or the whole clip when it has only one. Scanning the
    // packets once is cheap next to the encode itself and does not depend on a container field
    // that many mp4s leave unset.
    int64_t first_key = -1;
    int64_t second_key = -1;
    int64_t packets = 0;
    AVPacket* pkt = av_packet_alloc();
    while (pkt && av_read_frame(ctx, pkt) >= 0)
    {
      if (pkt->stream_index == video)
      {
        ++packets;
        if (pkt->flags & AV_PKT_FLAG_KEY)
        {
          if (first_key < 0) first_key = packets - 1;
          else if (second_key < 0) { second_key = packets - 1; av_packet_unref(pkt); break; }
        }
      }
      av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    if (second_key > first_key)
      settings.gop = static_cast<int32_t>(second_key - first_key);
    else if (packets > 0)
      settings.gop = static_cast<int32_t>(packets);
    // else leave 0 and let the encoder's two-second default stand -- an empty file has nothing
    // to match.
  }

  avformat_close_input(&ctx);
  log()->info("roi job: matching source -> codec={} gop={}", settings.codec, settings.gop);
}

bool run_roi_file_job(const roi_job_config& config)
{
  roi_job_config cfg = config;
  match_source_encode_defaults(cfg.encode, cfg.input_path);

  pipeline_config pcfg;
  pcfg.watch_id = "roi-file-job";
  pcfg.classes = cfg.classes;
  pcfg.min_confidence = cfg.min_confidence;
  pcfg.attach_debug_crops = false;
  pcfg.reid_embed_interval_sec = cfg.reid_embed_interval_sec;

  pipeline detector(pcfg, cfg.model_dir);
  roi_stream_client client;

  auto sink = std::make_shared<roi_encode_sink>(cfg, detector, client);

  std::mutex done_mutex;
  std::condition_variable done_cv;
  bool done = false;

  // Read as fast as the sink accepts and stop at the end rather than looping: this is one pass
  // over one clip, and looping would be an infinite transcode.
  auto source = std::make_shared<file_reader>("roi-file-job", /*loop=*/false, /*realtime=*/false,
    [&] {
      std::scoped_lock lock(done_mutex);
      done = true;
      done_cv.notify_all();
    });

  if (!source->open(cfg.input_path))
  {
    log()->error("roi job: cannot open {}", cfg.input_path);
    return false;
  }

  source->add_sink(sink);
  source->start();

  {
    std::unique_lock lock(done_mutex);
    done_cv.wait(lock, [&] { return done; });
  }
  source->stop();

  sink->flush();

  if (sink->failed())
  {
    log()->error("roi job: failed after {} frames", sink->frames_sent());
    return false;
  }
  if (sink->frames_sent() == 0)
  {
    log()->error("roi job: no frames were decoded from {}", cfg.input_path);
    return false;
  }

  roi_encode_report report;
  std::string error;
  if (!client.finish(report, error))
  {
    log()->error("roi job: the encoder ended badly: {}", error);
    return false;
  }

  log()->info("roi job: {} frames sent, {} detection passes, {} packets, {} bytes written{}",
    sink->frames_sent(), sink->detections_run(), report.packets, report.bytes,
    report.regions_dropped > 0
      ? std::string(", ") + std::to_string(report.regions_dropped) + " regions dropped"
      : std::string());

  if (!cfg.boxes_json_path.empty())
  {
    if (!sink->regions().write(cfg.boxes_json_path))
    {
      log()->error("roi job: cannot write {}", cfg.boxes_json_path);
      return false;
    }
    log()->info("roi job: regions written to {}", cfg.boxes_json_path);
  }

  if (!cfg.qpmap_json_path.empty())
  {
    if (!sink->qp_maps().write(cfg.qpmap_json_path))
    {
      log()->error("roi job: cannot write {}", cfg.qpmap_json_path);
      return false;
    }
    log()->info("roi job: qp_map written to {}", cfg.qpmap_json_path);
  }

  return true;
}

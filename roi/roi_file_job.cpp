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
#include "inference/pipeline.h"
#include "logging.h"
#include "reader/file_reader.h"
#include "roi/file_analyzer.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace {

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

// Takes each frame the analyzer hands on and pushes it to the encoder with the regions in force
// at that moment, or -- with detect_only -- only records them.
//
// Everything here runs on the reader's thread, one packet at a time. That is what removes the
// need for a queue: the reader cannot get ahead of the encoder because it is the encoder's call
// stack that it is waiting on.
class roi_encode_sink {
public:
  roi_encode_sink(const roi_job_config& config, roi_stream_client& client)
    : m_config(config)
    , m_client(client)
  {
    m_sidecar.input = config.input_path;
    m_sidecar.output = config.encode.output_path;
    m_qpmap.input = config.input_path;
    m_qpmap.output = config.encode.output_path;
  }

  const sidecar& regions() const { return m_sidecar; }
  const qpmap_sidecar& qp_maps() const { return m_qpmap; }

  bool handle_frame(file_analyzer& analyzer, const analyzed_frame& af)
  {
    const AVFrame* frame = af.frame;
    if (!ensure_encoder_open(frame, af)) return false;

    const double t = af.t;
    m_boxes = af.boxes;

    // Without an encoder the frame is already spent: the detector has had it, the regions are
    // recorded below, and YUV420P is the wire's format rather than the sidecar's.
    if (m_config.detect_only)
    {
      m_sidecar.frames.push_back(sidecar_frame{ t, *af.boxes });
      // Progress, because this is the one mode with nothing on the other end to report it: a
      // minute of silence over a long clip cannot be told from a hang. Once per second of
      // content, measured on the file's own timeline like everything else here.
      constexpr double kProgressIntervalSec = 1.0;
      if (t + 1e-9 >= m_next_progress_at)
      {
        log()->info("roi job: detected through t={:.2f} ({} frames, {} passes)", t,
          af.index + 1, analyzer.detections_run());
        m_next_progress_at = t + kProgressIntervalSec;
      }
      return true;
    }

    const std::vector<uint8_t>* yuv = analyzer.pack_yuv420p(frame);
    if (!yuv) return false;

    const int64_t pts = af.pts;
    bool sent = false;
    if (m_config.regions == roi_region_form::mask)
    {
      build_mask(frame->width, frame->height);
      sent = m_client.send_frame_mask(pts, yuv->data(), yuv->size(),
        m_mask.data(), m_mask_width, m_mask_height);
    }
    else if (m_config.regions == roi_region_form::qp_map)
    {
      build_qp_map(frame->width, frame->height);
      sent = m_client.send_frame_qp_map(pts, yuv->data(), yuv->size(),
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
      sent = m_client.send_frame(pts, yuv->data(), yuv->size(), *af.boxes);
    }

    if (!sent)
    {
      log()->error("roi job: the encoder stopped accepting frames after {}", af.index);
      return false;
    }

    m_sidecar.frames.push_back(sidecar_frame{ t, *af.boxes });
    return true;
  }

private:
  bool ensure_encoder_open(const AVFrame* frame, const analyzed_frame& af)
  {
    if (m_opened) return true;

    m_sidecar.width = frame->width;
    m_sidecar.height = frame->height;
    m_sidecar.fps = af.frame_rate.den > 0
      ? static_cast<double>(af.frame_rate.num) / af.frame_rate.den
      : 0.0;

    if (m_config.detect_only)
    {
      m_opened = true;
      log()->info("roi job: detecting only, {}x{} @ {:.2f} fps -> {}", frame->width,
        frame->height, m_sidecar.fps, m_config.boxes_json_path);
      return true;
    }

    std::string error;
    if (!m_client.open(m_config.encode, frame->width, frame->height,
          af.time_base.num, af.time_base.den, af.frame_rate.num, af.frame_rate.den,
          AV_PIX_FMT_YUV420P, error))
    {
      log()->error("roi job: encoder refused the stream: {}", error);
      return false;
    }
    m_opened = true;
    log()->info("roi job: encoding {}x{} @ {:.2f} fps -> {}", frame->width, frame->height,
      m_sidecar.fps, m_config.encode.output_path);
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
    // Match the encoder library's capabilities for libx264 / libx265.
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

    for (const auto& b : *m_boxes)
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
    for (const auto& b : *m_boxes)
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

    for (const auto& b : *m_boxes)
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

  const roi_job_config& m_config;
  roi_stream_client& m_client;

  // The regions of the frame being sent; owned by the analyzer.
  const std::vector<roi_box>* m_boxes = nullptr;

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
  double m_next_progress_at = 0;

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
  // Reads the source to fill in codec and gop, which only the encoder uses.
  if (!cfg.detect_only)
    match_source_encode_defaults(cfg.encode, cfg.input_path);

  pipeline_config pcfg;
  pcfg.watch_id = "roi-file-job";
  pcfg.classes = cfg.classes;
  pcfg.min_confidence = cfg.min_confidence;
  pcfg.attach_debug_crops = false;
  pcfg.reid_embed_interval_sec = cfg.reid_embed_interval_sec;

  pipeline detector(pcfg, cfg.model_dir);
  roi_stream_client client;

  roi_encode_sink encode(cfg, client);
  file_analyzer_config acfg;
  acfg.detect_fps = cfg.detect_fps;
  auto sink = std::make_shared<file_analyzer>(acfg, &detector,
    [&encode](file_analyzer& analyzer, const analyzed_frame& frame) {
      return encode.handle_frame(analyzer, frame);
    });

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
    log()->error("roi job: failed after {} frames", sink->frames());
    return false;
  }
  if (sink->frames() == 0)
  {
    log()->error("roi job: no frames were decoded from {}", cfg.input_path);
    return false;
  }

  if (cfg.detect_only)
  {
    // The gated count belongs in this line rather than in a debug log: a gated pass emits
    // nothing, so "few detections" and "few frames looked at" are different diagnoses and
    // this is the only place that can tell them apart.
    log()->info("roi job: {} frames decoded, {} detection passes, {} gated by motion, "
      "nothing encoded", sink->frames(), sink->detections_run(), detector.gated_frames());
  }
  else
  {
    roi_encode_report report;
    std::string error;
    if (!client.finish(report, error))
    {
      log()->error("roi job: the encoder ended badly: {}", error);
      return false;
    }

    log()->info("roi job: {} frames sent, {} detection passes, {} packets, {} bytes written{}",
      sink->frames(), sink->detections_run(), report.packets, report.bytes,
      report.regions_dropped > 0
        ? std::string(", ") + std::to_string(report.regions_dropped) + " regions dropped"
        : std::string());
  }

  if (!cfg.boxes_json_path.empty())
  {
    if (!encode.regions().write(cfg.boxes_json_path))
    {
      log()->error("roi job: cannot write {}", cfg.boxes_json_path);
      return false;
    }
    log()->info("roi job: regions written to {}", cfg.boxes_json_path);
  }

  if (!cfg.qpmap_json_path.empty())
  {
    if (!encode.qp_maps().write(cfg.qpmap_json_path))
    {
      log()->error("roi job: cannot write {}", cfg.qpmap_json_path);
      return false;
    }
    log()->info("roi job: qp_map written to {}", cfg.qpmap_json_path);
  }

  return true;
}

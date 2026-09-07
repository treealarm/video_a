#include "frame_sampler.h"
#include "logging.h"

#include <algorithm>
#include <span>
#include <vector>

namespace {
// sve::PixelFormat back to the one swscale names. The library keeps its own enumeration so that
// its public headers name no FFmpeg type; swscale is an FFmpeg call, so the translation has to
// happen somewhere, and here is the only place this program makes it.
//
// The JPEG-range formats (YUVJ*) are not in the list because they never arrive: the decoder
// normalises them to their limited-range twin and reports the range separately, through
// DecodedFrameInfo::full_range. That is the same normalisation this file used to do itself --
// feeding a YUVJ format to sws_getContext logs "deprecated pixel format used" on every frame --
// only now it is done once, for every consumer of the library.
AVPixelFormat to_av_pixel_format(sve::PixelFormat format)
{
  switch (format)
  {
    case sve::PixelFormat::YUV420P: return AV_PIX_FMT_YUV420P;
    case sve::PixelFormat::NV12:    return AV_PIX_FMT_NV12;
    case sve::PixelFormat::YUV422P: return AV_PIX_FMT_YUV422P;
    case sve::PixelFormat::YUV444P: return AV_PIX_FMT_YUV444P;
    case sve::PixelFormat::GRAY8:   return AV_PIX_FMT_GRAY8;
    case sve::PixelFormat::P010:    return AV_PIX_FMT_P010;
    // A Main10 source decoded in software. It reaches here whenever the hardware has no
    // ten-bit path for the codec and the decoder falls back, which is the case this whole
    // switch has to survive: an unmapped format discards every frame of the stream.
    case sve::PixelFormat::YUV420P10: return AV_PIX_FMT_YUV420P10;
    default:                        return AV_PIX_FMT_NONE;
  }
}

sve::InputCodec to_input_codec(AVCodecID id)
{
  switch (id)
  {
    case AV_CODEC_ID_H264: return sve::InputCodec::H264;
    case AV_CODEC_ID_HEVC: return sve::InputCodec::H265;
    case AV_CODEC_ID_MJPEG: return sve::InputCodec::Jpeg;
    default: return sve::InputCodec::Raw;
  }
}
}

frame_sampler::frame_sampler(std::function<void(const decoded_frame&)> callback, uint32_t sample_fps,
  sve::DecodeDevice device)
  : m_callback(std::move(callback))
  , m_emit_period(std::chrono::milliseconds(1000 / (sample_fps > 0 ? sample_fps : 1)))
  , m_device(std::move(device))
{
  m_worker = std::thread([this] { worker_loop(); });
}

frame_sampler::~frame_sampler()
{
  m_running = false;
  m_cv.notify_all();
  if (m_worker.joinable())
    m_worker.join();
}

void frame_sampler::on_packet(const std::shared_ptr<media_packet>& pkt)
{
  if (!pkt || !pkt->packet) return;
  if (pkt->media_type != AVMEDIA_TYPE_VIDEO) return;

  const bool is_key = (pkt->packet->flags & AV_PKT_FLAG_KEY) != 0;

  // Keyframes are timed whatever the mode, because the measurement is what chooses the mode.
  if (is_key)
  {
    note_keyframe();
    update_mode();
  }

  if (!m_full_decode && !is_key)
    return;  // the cheap path: non-key packets never reach the decoder at all

  // Enqueue only — the heavy decode + inference happens on the worker thread so this (the RTSP read
  // loop) returns immediately and keeps draining the socket. Blocking here would stall the source
  // (see the header note on live-video stutter), so the queue sheds instead.
  {
    std::scoped_lock lock(m_queue_mutex);
    if (!m_full_decode)
    {
      while (m_queue.size() >= k_max_queue)
        m_queue.pop_front();
    }
    else if (m_queue.size() >= k_max_queue_full)
    {
      // Shed a whole GOP rather than a packet. Dropping one P-frame would leave the decoder
      // producing garbage until the next keyframe, which is worse than losing the second or two
      // this discards: pop the front, then keep popping until the queue starts on a keyframe
      // again — a packet the decoder can actually begin from.
      m_queue.pop_front();
      while (!m_queue.empty() && !(m_queue.front().pkt->packet->flags & AV_PKT_FLAG_KEY))
        m_queue.pop_front();
      log()->warn("frame_sampler: decode is behind, dropped a GOP");
    }
    // Stamped here, on the read loop, rather than after the decode: this is the last moment that
    // still corresponds to the picture rather than to how busy we are.
    m_queue.push_back({pkt, std::chrono::system_clock::now()});
  }
  m_cv.notify_one();
}

void frame_sampler::note_keyframe()
{
  const auto now = std::chrono::steady_clock::now();
  m_keyframe_times.push_back(now);
  while (m_keyframe_times.size() > k_interval_gaps + 1)
    m_keyframe_times.pop_front();

  if (m_keyframe_times.size() < 2)
    return;

  // Median of the gaps, not the mean. The outliers here are one-sided and large -- a reconnect or
  // a stalled publisher contributes one enormous gap -- and a mean would let a single one of them
  // push a well-behaved stream onto the expensive path and keep it there.
  std::vector<std::chrono::milliseconds> gaps;
  gaps.reserve(m_keyframe_times.size() - 1);
  for (size_t i = 1; i < m_keyframe_times.size(); ++i)
    gaps.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
      m_keyframe_times[i] - m_keyframe_times[i - 1]));
  std::sort(gaps.begin(), gaps.end());
  m_keyframe_interval = gaps[gaps.size() / 2];
}

void frame_sampler::update_mode()
{
  if (m_keyframe_interval.count() == 0)
    return;  // nothing measured yet; the cheap path is also the one we would have chosen

  // Hysteresis, because the interesting case is a stream whose keyframe spacing sits right at the
  // requested period: without a band it would flip modes on jitter alone, and each flip costs a
  // decoder that has to resynchronise.
  const auto enter = m_emit_period * 3 / 2;
  const auto leave = m_emit_period * 11 / 10;

  const bool want_full = m_full_decode ? (m_keyframe_interval > leave)
                                       : (m_keyframe_interval > enter);
  if (want_full == m_full_decode)
    return;

  m_full_decode = want_full;
  log()->info("frame_sampler: keyframe interval {} ms vs sample period {} ms — {} decode",
    m_keyframe_interval.count(), m_emit_period.count(),
    m_full_decode ? "switching to full" : "back to keyframe-only");
}

void frame_sampler::worker_loop()
{
  while (m_running)
  {
    queued_packet queued;
    {
      std::unique_lock lock(m_queue_mutex);
      m_cv.wait(lock, [&] { return !m_queue.empty() || !m_running; });
      if (!m_running)
        break;
      queued = std::move(m_queue.front());
      m_queue.pop_front();
    }
    if (queued.pkt)
      decode_packet(*queued.pkt, queued.arrived_at);
  }
}

void frame_sampler::decode_packet(
  const media_packet& pkt, std::chrono::system_clock::time_point arrived_at)
{
  if (!ensure_decoder(pkt)) return;

  m_current_arrival = arrived_at;

  // The timestamp is handed over as the packet carries it. Nothing here reads it back -- frames
  // are dated by m_current_arrival, for the reason that member documents -- but a decoder that is
  // fed timestamps reorders correctly, and one that is fed nothing has to guess.
  const sve::MediaTimestamp ts{ pkt.packet->pts, { pkt.time_base.num, pkt.time_base.den } };
  const sve::Status status = m_decoder.Push(
    { pkt.packet->data, static_cast<size_t>(pkt.packet->size) }, ts,
    [this](const sve::Frame& frame, const sve::DecodedFrameInfo& info)
    {
      handle_decoded_frame(frame, info);
    });
  if (!status)
    log()->warn("frame_sampler: decode failed: {}", status.message());
}

bool frame_sampler::ensure_decoder(const media_packet& pkt)
{
  if (m_decoder_open) return true;
  if (!pkt.codec_parameters) return false;

  const sve::InputCodec codec = to_input_codec(pkt.codec_parameters->codec_id);
  if (codec == sve::InputCodec::Raw)
  {
    log()->error("frame_sampler: no decoder for codec_id={}",
      static_cast<int>(pkt.codec_parameters->codec_id));
    return false;
  }

  const std::span<const uint8_t> extradata(
    pkt.codec_parameters->extradata,
    pkt.codec_parameters->extradata ? static_cast<size_t>(pkt.codec_parameters->extradata_size) : 0);

  const sve::Status status = m_decoder.Open(codec, extradata, m_device);
  if (!status)
  {
    log()->error("frame_sampler: decoder would not open: {}", status.message());
    return false;
  }

  m_decoder_open = true;
  return true;
}

bool frame_sampler::ensure_sws_context(
  int width, int height, sve::PixelFormat format, bool full_range)
{
  if (m_sws_ctx && width == m_sws_width && height == m_sws_height && format == m_sws_format
      && full_range == m_sws_full_range)
    return true;

  const AVPixelFormat src_format = to_av_pixel_format(format);
  if (src_format == AV_PIX_FMT_NONE)
  {
    log()->error("frame_sampler: cannot convert {} to BGR24", sve::pixel_format_name(format));
    return false;
  }

  m_sws_ctx.reset(sws_getContext(
    width, height, src_format,
    width, height, AV_PIX_FMT_BGR24,
    SWS_BILINEAR, nullptr, nullptr, nullptr));

  if (!m_sws_ctx) return false;

  if (full_range)
  {
    // JPEG full-range (0..255) input — tell sws so luma/chroma aren't wrongly rescaled.
    const int* coeffs = sws_getCoefficients(SWS_CS_ITU601);
    sws_setColorspaceDetails(m_sws_ctx.get(), coeffs, /*srcRange=*/1, coeffs, /*dstRange=*/1, 0, 1 << 16, 1 << 16);
  }

  m_sws_width = width;
  m_sws_height = height;
  m_sws_format = format;
  m_sws_full_range = full_range;
  return true;
}

void frame_sampler::handle_decoded_frame(
  const sve::Frame& frame, const sve::DecodedFrameInfo& info)
{
  if (!m_callback) return;
  if (!frame) return;

  if (!m_path_reported)
  {
    m_path_reported = true;
    log()->info("frame_sampler: decoding {} on {}", m_decoder.decoder_name(),
      m_decoder.hardware_name().empty() ? "cpu" : m_decoder.hardware_name());
  }

  // The schedule, applied in both modes so that sample_fps means the same thing whichever one is
  // running -- in full decode it is what makes most decoded frames free, and in keyframe-only it
  // stops a stream that keys faster than asked from running inference more often than asked.
  //
  // Tested before anything else in this function, and on a hardware device that ordering is worth
  // more than it used to be: everything below -- the transfer back from the GPU as well as the
  // colour conversion -- is skipped for a frame the schedule does not want, so a stream decoded in
  // full at 25 frames a second crosses the bus once a second.
  //
  // The tolerance matters more than it looks. A camera keying once a second against a one-second
  // period lands a few milliseconds early as often as late, and a bare `>= period` would reject
  // every early arrival and halve the rate to one frame every two seconds.
  const auto now = std::chrono::steady_clock::now();
  if (m_next_emit.time_since_epoch().count() == 0)
    m_next_emit = now;  // first frame is always due
  if (now < m_next_emit - m_emit_period / 10)
    return;
  // Advanced from the deadline rather than from this frame, so accepting an early one does not
  // pull the whole schedule forward with it. Measuring from the last emit instead made every
  // interval come out a tolerance short, which over a run is a tenth more inference than asked.
  m_next_emit += m_emit_period;
  if (m_next_emit < now)
    m_next_emit = now + m_emit_period;  // fell behind (a stall, a reconnect) -- resync rather than burst

  // Into system memory, if it is not there already. A frame decoded on the CPU comes back
  // unchanged and costs nothing here.
  sve::Result<sve::Frame> readable = m_decoder.Download(frame);
  if (!readable)
  {
    log()->warn("frame_sampler: {}", readable.status().message());
    return;
  }
  sve::IFrameBuffer& pixels = readable.value().mutable_buffer();
  if (pixels.width() <= 0 || pixels.height() <= 0) return;

  if (!ensure_sws_context(pixels.width(), pixels.height(), pixels.format(), info.full_range))
    return;

  decoded_frame out;
  out.width = pixels.width();
  out.height = pixels.height();
  out.bgr.resize(static_cast<size_t>(out.width) * static_cast<size_t>(out.height) * 3);
  // The packet's arrival, not the clock now: see m_current_arrival. Falls back to now() only if a
  // frame somehow reaches here without a packet having been decoded, which nothing does today.
  out.captured_at = m_current_arrival.time_since_epoch().count() != 0
    ? m_current_arrival
    : std::chrono::system_clock::now();

  const uint8_t* src_data[4] = {};
  int src_linesize[4] = {};
  for (int i = 0; i < 4; ++i)
  {
    const sve::PlaneView plane = pixels.plane(i);
    src_data[i] = plane.empty() ? nullptr : plane.bytes.data();
    src_linesize[i] = plane.stride;
  }

  uint8_t* dst_data[4] = { out.bgr.data(), nullptr, nullptr, nullptr };
  int dst_linesize[4] = { out.width * 3, 0, 0, 0 };

  sws_scale(m_sws_ctx.get(), src_data, src_linesize, 0, out.height, dst_data, dst_linesize);

  m_callback(out);
}

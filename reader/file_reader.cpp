#include <chrono>
#include <thread>

#include "interfaces/av_deleters.h"
#include "file_reader.h"
#include "logging.h"

extern "C" {
#include <libavutil/time.h>
}

file_reader::file_reader(std::string watch_id, bool loop, bool realtime,
  std::function<void()> on_end)
  : m_loop(loop)
  , m_realtime(realtime)
  , m_on_end(std::move(on_end))
  , m_watch_id(std::move(watch_id))
{
}

file_reader::~file_reader()
{
  stop();
  disconnect();
}

bool file_reader::open(const std::string& path)
{
  m_path = path;
  return !m_path.empty();
}

bool file_reader::connect()
{
  disconnect();

  if (m_path.empty())
  {
    log()->error("file_reader: empty path [watch={}]", m_watch_id);
    return false;
  }

  AVFormatContext* ctx = avformat_alloc_context();
  if (!ctx)
    return false;

  // The same interrupt callback rtsp_reader installs, for the same reason: a stop must be able to
  // end a blocking read rather than wait for it.
  ctx->interrupt_callback.callback = [](void* opaque) -> int {
    return static_cast<const file_reader*>(opaque)->m_running ? 0 : 1;
  };
  ctx->interrupt_callback.opaque = this;

  if (avformat_open_input(&ctx, m_path.c_str(), nullptr, nullptr) < 0)
  {
    log()->error("file_reader: cannot open [watch={}]: {}", m_watch_id, m_path);
    avformat_close_input(&ctx);
    return false;
  }

  m_ctx.reset(ctx);

  if (avformat_find_stream_info(m_ctx.get(), nullptr) < 0)
  {
    log()->error("file_reader: no stream info [watch={}]: {}", m_watch_id, m_path);
    disconnect();
    return false;
  }

  m_video_index = -1;
  m_forward_stream_indexes.clear();
  for (unsigned i = 0; i < m_ctx->nb_streams; ++i)
  {
    const AVMediaType media_type = m_ctx->streams[i]->codecpar->codec_type;
    if (media_type == AVMEDIA_TYPE_VIDEO && m_video_index < 0)
      m_video_index = static_cast<int>(i);

    if (media_type == AVMEDIA_TYPE_VIDEO || media_type == AVMEDIA_TYPE_AUDIO)
      m_forward_stream_indexes.insert(static_cast<int>(i));
  }

  if (m_video_index < 0)
  {
    log()->error("file_reader: no video stream [watch={}]: {}", m_watch_id, m_path);
    disconnect();
    return false;
  }

  m_pace_origin = std::chrono::steady_clock::now();
  m_pace_first_pts_us = AV_NOPTS_VALUE;

  log()->info("file_reader: opened [watch={}]: {} ({}x{})", m_watch_id, m_path,
    m_ctx->streams[m_video_index]->codecpar->width,
    m_ctx->streams[m_video_index]->codecpar->height);
  return true;
}

void file_reader::disconnect()
{
  m_video_index = -1;
  m_forward_stream_indexes.clear();
  m_ctx.reset();
}

void file_reader::add_sink(std::shared_ptr<media_sink> sink)
{
  m_sinks_helper.add_sink(sink);
}

void file_reader::remove_sink(std::shared_ptr<media_sink> sink)
{
  m_sinks_helper.remove_sink(sink);
}

AVStream* file_reader::video_stream() const
{
  if (m_ctx && m_video_index >= 0)
    return m_ctx->streams[m_video_index];
  return nullptr;
}

void file_reader::start()
{
  if (m_running)
    return;
  m_running = true;
  m_thread = std::thread([this] { read_loop(); });
}

void file_reader::stop()
{
  m_running = false;
  if (m_thread.joinable())
    m_thread.join();
}

void file_reader::pace(const media_packet& pkt)
{
  if (!m_realtime)
    return;
  if (!pkt.packet || pkt.packet->pts == AV_NOPTS_VALUE)
    return;

  const int64_t pts_us = av_rescale_q(pkt.packet->pts, pkt.time_base, AVRational{ 1, 1000000 });
  if (m_pace_first_pts_us == AV_NOPTS_VALUE)
  {
    m_pace_first_pts_us = pts_us;
    return;
  }

  const auto due = m_pace_origin + std::chrono::microseconds(pts_us - m_pace_first_pts_us);
  const auto now = std::chrono::steady_clock::now();
  if (due > now)
    std::this_thread::sleep_for(due - now);
}

void file_reader::read_loop()
{
  while (m_running)
  {
    if (!m_ctx || m_video_index < 0)
    {
      if (!connect())
        break;  // a file that cannot be opened will not open on a retry either
    }

    avpacket_ptr pkt(av_packet_alloc(), avpacket_deleter{});
    if (av_read_frame(m_ctx.get(), pkt.get()) < 0)
    {
      // End of input. Looping restarts the pass and resets the pacing anchor with it, so the
      // second pass is timed from its own beginning rather than from the first frame ever read.
      if (m_loop && m_running)
      {
        if (av_seek_frame(m_ctx.get(), m_video_index, 0, AVSEEK_FLAG_BACKWARD) < 0)
          break;
        m_pace_origin = std::chrono::steady_clock::now();
        m_pace_first_pts_us = AV_NOPTS_VALUE;
        continue;
      }
      break;
    }

    if (!m_forward_stream_indexes.contains(pkt->stream_index))
      continue;

    auto media = std::make_shared<media_packet>();
    media->packet = pkt;
    media->stream_index = pkt->stream_index;
    media->time_base = m_ctx->streams[pkt->stream_index]->time_base;
    media->media_type = m_ctx->streams[pkt->stream_index]->codecpar->codec_type;
    media->frame_rate = m_ctx->streams[pkt->stream_index]->avg_frame_rate;
    media->sample_rate = m_ctx->streams[pkt->stream_index]->codecpar->sample_rate;
    media->nb_samples = 0;

    avcodec_parameters_ptr codec_copy(avcodec_parameters_alloc(),
      [](AVCodecParameters* params) { avcodec_parameters_free(&params); });
    if (!codec_copy)
      continue;
    if (avcodec_parameters_copy(codec_copy.get(),
          m_ctx->streams[pkt->stream_index]->codecpar) < 0)
      continue;
    media->codec_parameters = std::move(codec_copy);

    pace(*media);

    for (auto& s : m_sinks_helper.snapshot_sinks())
      s->on_packet(media);
  }

  const bool reached_end = m_running;
  disconnect();

  // Only an exhausted input reports an end; a stop() is the caller's own doing and needs no
  // notification back.
  if (reached_end && m_on_end)
    m_on_end();
}

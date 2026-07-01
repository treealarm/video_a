#include <chrono>
#include <format>
#include <string_view>
#include <thread>
#include "interfaces/av_deleters.h"
#include "rtsp_reader.h"
#include "logging.h"

namespace {
constexpr auto kWatchdogInterval = std::chrono::seconds(5);

constexpr std::string_view kRtspPrefix = "rtsp://";

// True if the URL already carries userinfo (user[:pass]@) in its authority,
// e.g. rtsp://user:pass@host/path. In that case the stream provides its own
// credentials and we must not override them with the watch's.
bool rtsp_url_has_credentials(const std::string& url)
{
  if (url.rfind(kRtspPrefix, 0) != 0) return false;

  const std::string_view rest(url.data() + kRtspPrefix.size(),
    url.size() - kRtspPrefix.size());
  const auto at = rest.find('@');
  if (at == std::string_view::npos) return false;

  // userinfo must appear in the authority, before the path/query starts.
  const auto authority_end = rest.find_first_of("/?");
  return authority_end == std::string_view::npos || at < authority_end;
}

std::string build_rtsp_url_with_auth(const std::string& url,
  const std::string& user,
  const std::string& pass)
{
  if (user.empty()) return url;

  if (url.rfind(kRtspPrefix, 0) != 0) return url;

  return std::string(kRtspPrefix) + user + ":" + pass + "@"
    + url.substr(kRtspPrefix.size());
}
}

rtsp_reader::rtsp_reader(
  const std::string& watch_id, const std::string& user, const std::string& pass)
  : m_watch_id(watch_id)
  , m_user(user)
  , m_password(pass)
{
}

rtsp_reader::~rtsp_reader()
{
  stop();
  disconnect();
}

bool rtsp_reader::open(const std::string& url)
{
  m_rtsp_url = url;
  return true;
}

bool rtsp_reader::connect()
{
  disconnect();

  if (m_rtsp_url.empty())
  {
    log()->error("rtsp_reader: empty rtsp_url [watch={}]", m_watch_id);
    return false;
  }

  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "rtsp_transport", "tcp", 0);
  av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);
  av_dict_set(&opts, "timeout", "5000000", 0);
  av_dict_set(&opts, "rw_timeout", "5000000", 0);

  // If the stream URL already embeds credentials, honour them and skip the
  // watch-wide ones; otherwise fall back to the watch's user/password.
  const bool url_has_creds = rtsp_url_has_credentials(m_rtsp_url);

  if (!url_has_creds && !m_user.empty())
  {
    av_dict_set(&opts, "user", m_user.c_str(), 0);
    av_dict_set(&opts, "password", m_password.c_str(), 0);
  }
  AVFormatContext* ctx = nullptr;

  std::string final_url = url_has_creds
    ? m_rtsp_url
    : build_rtsp_url_with_auth(m_rtsp_url, m_user, m_password);

  if (avformat_open_input(&ctx, final_url.c_str(), nullptr, &opts) < 0)
  {
    log()->error("Failed to open RTSP [watch={}]: {}", m_watch_id, m_rtsp_url);
    av_dict_free(&opts);
    avformat_close_input(&ctx);
    return false;
  }

  m_ctx.reset(ctx);

  av_dict_free(&opts);

  if (avformat_find_stream_info(m_ctx.get(), nullptr) < 0)
  {
    log()->error("Failed to get stream info [watch={}]: {}", m_watch_id, m_rtsp_url);
    disconnect();
    return false;
  }

  m_video_index = -1;
  m_forward_stream_indexes.clear();
  for (unsigned i = 0; i < m_ctx->nb_streams; ++i)
  {
    const AVMediaType media_type = m_ctx->streams[i]->codecpar->codec_type;
    if (media_type == AVMEDIA_TYPE_VIDEO && m_video_index < 0)
    {
      m_video_index = static_cast<int>(i);
    }

    if (media_type == AVMEDIA_TYPE_VIDEO || media_type == AVMEDIA_TYPE_AUDIO)
    {
      m_forward_stream_indexes.insert(static_cast<int>(i));
    }
  }

  if (m_video_index < 0)
  {
    log()->error("No video stream [watch={}]: {}", m_watch_id, m_rtsp_url);
    disconnect();
    return false;
  }

  log()->info("RTSP connected [watch={}]: {}", m_watch_id, m_rtsp_url);
  return true;
}

void rtsp_reader::disconnect() {
  m_video_index = -1;
  m_forward_stream_indexes.clear();
  m_ctx.reset();
}

void rtsp_reader::add_sink(std::shared_ptr<media_sink> sink)
{
  m_sinks_helper.add_sink(sink);
}

void rtsp_reader::remove_sink(std::shared_ptr<media_sink> sink)
{
  m_sinks_helper.remove_sink(sink);
}

AVStream* rtsp_reader::video_stream() const
{
  if (m_ctx && m_video_index >= 0)
    return m_ctx->streams[m_video_index];
  return nullptr;
}

void rtsp_reader::start()
{
  if (m_running)
    return;
  m_running = true;
  m_thread = std::thread([this] { read_loop(); });
}

void rtsp_reader::stop()
{
  m_running = false;
  if (m_thread.joinable())
    m_thread.join();
}

void rtsp_reader::read_loop()
{
  auto last_frame_at = std::chrono::steady_clock::now();

  while (m_running) {
    if (!m_ctx || m_video_index < 0 || m_forward_stream_indexes.empty())
    {
      if (!connect())
      {
        std::this_thread::sleep_for(kWatchdogInterval);
        continue;
      }
      last_frame_at = std::chrono::steady_clock::now();
    }

    avpacket_ptr pkt(av_packet_alloc(), avpacket_deleter{});
    const int read_result = av_read_frame(m_ctx.get(), pkt.get());

    if (read_result < 0)
    {
      const auto now = std::chrono::steady_clock::now();
      if (now - last_frame_at >= kWatchdogInterval)
      {
        log()->warn("RTSP watchdog: reconnecting [watch={}]: {}", m_watch_id, m_rtsp_url);
        disconnect();
      }
      continue;
    }

    if (!m_forward_stream_indexes.contains(pkt->stream_index))
      continue;

    last_frame_at = std::chrono::steady_clock::now();

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
    if (!codec_copy) {
      continue;
    }

    if (avcodec_parameters_copy(codec_copy.get(), m_ctx->streams[pkt->stream_index]->codecpar) < 0)
    {
      continue;
    }

    media->codec_parameters = std::move(codec_copy);

    for (auto& s : m_sinks_helper.snapshot_sinks())
    {
      s->on_packet(media);
    }
  }

  disconnect();
}

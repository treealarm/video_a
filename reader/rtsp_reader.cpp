#include <cerrno>
#include <chrono>
#include <format>
#include <string_view>
#include <thread>
#include "interfaces/av_deleters.h"
#include "rtsp_reader.h"
#include "logging.h"

namespace {
constexpr auto kWatchdogInterval = std::chrono::seconds(5);

// Pause between two failed reads of a connection that is still considered live. Without it a
// stream that keeps failing is read in a tight loop that burns a core and, because libavformat
// logs every one of those failures, buries the log under thousands of identical lines per second.
// Short enough to be invisible next to a frame interval on any real stream.
constexpr auto kReadRetryInterval = std::chrono::milliseconds(20);

// Pause after the connection was torn down, before opening the next one. A camera that accepts an
// RTSP session and drops it immediately would otherwise be reconnected as fast as the handshake
// completes; the connect-failed path below already backs off, this is the same courtesy for the
// case where connecting is exactly what keeps succeeding.
constexpr auto kReconnectInterval = std::chrono::seconds(1);

// Errors that say the far end is gone rather than that this read did not work out. Waiting the
// full watchdog interval for one of these buys nothing -- no further packet is coming on this
// connection -- so they reconnect at once.
bool is_connection_lost(int averror)
{
  return averror == AVERROR_EOF
    || averror == AVERROR(EPIPE)
    || averror == AVERROR(ECONNRESET)
    || averror == AVERROR(ENOTCONN)
    || averror == AVERROR(ETIMEDOUT);
}

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
  const std::string& watch_id, const std::string& user, const std::string& pass, bool listen)
  : m_listen(listen)
  , m_watch_id(watch_id)
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
  if (m_listen)
  {
    // Accept a publish instead of making one, and wait as long as it takes.
    //
    // Not a bounded wait retried by the loop below, which is what this was first: a listening
    // socket that times out is not released for the next attempt, so every retry after the first
    // failed with the address already in use -- by us -- and the watch never started. One bind for
    // the lifetime of the watch has no such failure mode, and it also removes the window where a
    // producer that connects between two attempts is refused.
    //
    // Blocking is safe because the interrupt callback below ends the wait when the reader is
    // stopped; without it a watch could not be removed until something published to it.
    //
    // A day rather than the -1 that means "forever": the RTSP layer takes this in seconds and
    // hands the TCP layer milliseconds, so -1 arrives there as -1000 and is rejected as out of
    // range. A day is indistinguishable from forever within a session, and the interrupt callback
    // is what actually ends the wait.
    av_dict_set(&opts, "rtsp_flags", "listen", 0);
    av_dict_set(&opts, "listen_timeout", "86400", 0);
  }
  else
  {
    av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);
    av_dict_set(&opts, "timeout", "5000000", 0);
    av_dict_set(&opts, "rw_timeout", "5000000", 0);
  }

  // If the stream URL already embeds credentials, honour them and skip the
  // watch-wide ones; otherwise fall back to the watch's user/password.
  const bool url_has_creds = rtsp_url_has_credentials(m_rtsp_url);

  if (!url_has_creds && !m_user.empty())
  {
    av_dict_set(&opts, "user", m_user.c_str(), 0);
    av_dict_set(&opts, "password", m_password.c_str(), 0);
  }
  // Allocated here rather than by avformat_open_input, which is the only way to install an
  // interrupt callback that is already in place while the open itself blocks -- and in listen
  // mode the open is exactly what blocks.
  AVFormatContext* ctx = avformat_alloc_context();
  if (!ctx)
  {
    av_dict_free(&opts);
    return false;
  }
  ctx->interrupt_callback.callback = [](void* opaque) -> int {
    return static_cast<const rtsp_reader*>(opaque)->m_running ? 0 : 1;
  };
  ctx->interrupt_callback.opaque = this;

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
      const bool connection_lost = is_connection_lost(read_result);
      const auto now = std::chrono::steady_clock::now();
      if (connection_lost || now - last_frame_at >= kWatchdogInterval)
      {
        if (connection_lost)
          log()->warn("RTSP connection lost, reconnecting [watch={}]: {}", m_watch_id, m_rtsp_url);
        else
          log()->warn("RTSP watchdog: reconnecting [watch={}]: {}", m_watch_id, m_rtsp_url);
        disconnect();
        // The interrupt callback only ends a blocking libav call, not a sleep, so the backoff is
        // spent in short slices that re-check m_running -- a watch being removed must not have to
        // wait it out.
        for (auto waited = std::chrono::milliseconds::zero();
             m_running && waited < kReconnectInterval;
             waited += kReadRetryInterval)
        {
          std::this_thread::sleep_for(kReadRetryInterval);
        }
        continue;
      }
      // Still within the watchdog's grace period: the error may yet be transient (EAGAIN on a
      // non-blocking socket, a short read), so keep the connection and try again -- but not
      // immediately.
      std::this_thread::sleep_for(kReadRetryInterval);
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

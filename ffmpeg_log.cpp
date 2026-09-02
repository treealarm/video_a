#include "ffmpeg_log.h"

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

extern "C" {
#include <libavutil/log.h>
}

#include "logging.h"

namespace {
// How long a given message stays quiet after it has been logged once. Long enough that a stream
// stuck in a read-fail loop costs one line every few seconds instead of a screenful per second,
// short enough that the reconnect it precedes is still visibly attached to it.
constexpr auto kDebounceWindow = std::chrono::seconds(5);

// Distinct messages tracked at once. A bound rather than a growing map: the key is the formatted
// text, and FFmpeg interpolates frame numbers and timestamps into some of its messages, so an
// unbounded map is an unbounded leak on a long-lived worker.
constexpr size_t kMaxTrackedMessages = 256;

struct repeat_state
{
  std::chrono::steady_clock::time_point last_emit{};
  uint64_t suppressed = 0;
};

std::mutex g_mutex;
std::unordered_map<std::string, repeat_state> g_seen;

spdlog::level::level_enum to_spdlog_level(int av_level)
{
  if (av_level <= AV_LOG_FATAL)   return spdlog::level::critical;
  if (av_level <= AV_LOG_ERROR)   return spdlog::level::err;
  if (av_level <= AV_LOG_WARNING) return spdlog::level::warn;
  if (av_level <= AV_LOG_INFO)    return spdlog::level::info;
  if (av_level <= AV_LOG_VERBOSE) return spdlog::level::debug;
  return spdlog::level::trace;
}

// "rtsp", "h264", ... — what av_log's own "[rtsp @ 0x7ffe540c8240]" prefix is built from, minus
// the pointer. The pointer differs per demuxer instance, so keeping it would make the same
// message from two watches two distinct keys and defeat the whole point of the debounce.
const char* component_name(void* ptr)
{
  const AVClass* const* avc = static_cast<const AVClass* const*>(ptr);
  if (!avc || !*avc) return "ffmpeg";
  if ((*avc)->item_name)
  {
    if (const char* name = (*avc)->item_name(ptr)) return name;
  }
  return (*avc)->class_name ? (*avc)->class_name : "ffmpeg";
}

void drop_stale_entries(std::chrono::steady_clock::time_point now)
{
  for (auto it = g_seen.begin(); it != g_seen.end();)
  {
    // Entries still holding a suppressed count stay: dropping one loses the "+N more" that is the
    // only record those lines ever existed.
    if (it->second.suppressed == 0 && now - it->second.last_emit >= kDebounceWindow)
      it = g_seen.erase(it);
    else
      ++it;
  }
}

void ffmpeg_log_callback(void* ptr, int level, const char* fmt, va_list vl)
{
  if (level > av_log_get_level()) return;

  char text[1024];
  const int written = std::vsnprintf(text, sizeof(text), fmt, vl);
  if (written <= 0) return;

  std::string message(text);
  // FFmpeg messages carry their own newline, and a few are emitted in fragments that its default
  // callback stitches together; we log line-at-a-time and just trim the trailing whitespace.
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r' || message.back() == ' '))
    message.pop_back();
  if (message.empty()) return;

  const char* const component = component_name(ptr);
  std::string key = std::string(component) + '\x1f' + message;

  const auto now = std::chrono::steady_clock::now();
  uint64_t suppressed = 0;
  {
    const std::lock_guard<std::mutex> lock(g_mutex);

    if (g_seen.size() >= kMaxTrackedMessages)
      drop_stale_entries(now);

    auto [it, inserted] = g_seen.try_emplace(std::move(key));
    if (!inserted && now - it->second.last_emit < kDebounceWindow)
    {
      ++it->second.suppressed;
      return;
    }
    suppressed = std::exchange(it->second.suppressed, 0);
    it->second.last_emit = now;
  }

  const auto lvl = to_spdlog_level(level);
  if (suppressed > 0)
    log()->log(lvl, "[{}] {} (+{} suppressed)", component, message, suppressed);
  else
    log()->log(lvl, "[{}] {}", component, message);
}
}

void install_ffmpeg_log_bridge()
{
  av_log_set_callback(ffmpeg_log_callback);
}

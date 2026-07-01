#include "detection_queue.h"

#include "../logging.h"

void detection_queue::push(queued_detection item)
{
  std::scoped_lock lock(m_mutex);
  if (m_queue.size() >= kMaxQueueSize)
  {
    log()->warn("detection_queue: full ({} items), dropping oldest — no StreamDetections consumer reading fast enough", kMaxQueueSize);
    m_queue.pop_front(); // drop-oldest
  }

  m_queue.push_back(std::move(item));
  m_cv.notify_one();
}

std::optional<queued_detection> detection_queue::pop_wait(std::chrono::milliseconds timeout)
{
  std::unique_lock lock(m_mutex);
  if (!m_cv.wait_for(lock, timeout, [this] { return !m_queue.empty(); }))
    return std::nullopt;

  auto item = std::move(m_queue.front());
  m_queue.pop_front();
  return item;
}

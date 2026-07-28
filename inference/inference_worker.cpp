#include "inference_worker.h"

#include <exception>

#include "../logging.h"

inference_worker::inference_worker(std::function<void(const decoded_frame&)> detect)
  : m_detect(std::move(detect))
{
  m_worker = std::thread([this] { worker_loop(); });
}

inference_worker::~inference_worker()
{
  m_running = false;
  m_cv.notify_all();
  if (m_worker.joinable())
    m_worker.join();
}

void inference_worker::submit(const decoded_frame& frame)
{
  {
    std::scoped_lock lock(m_mutex);
    while (m_queue.size() >= k_max_queue)
      m_queue.pop_front();
    m_queue.push_back(frame);
  }
  m_cv.notify_one();
}

void inference_worker::worker_loop()
{
  while (m_running)
  {
    decoded_frame frame;
    {
      std::unique_lock lock(m_mutex);
      m_cv.wait(lock, [&] { return !m_queue.empty() || !m_running; });
      if (!m_running)
        break;
      frame = std::move(m_queue.front());
      m_queue.pop_front();
    }
    // Nothing sits above this std::thread to catch anything a detector throws, so an escaping
    // exception would std::terminate the whole process — every other watch included — over one
    // bad frame. Log and keep the stage alive instead.
    try
    {
      if (m_detect)
        m_detect(frame);
    }
    catch (const std::exception& ex)
    {
      log()->error("inference_worker: detect stage threw, dropping frame: {}", ex.what());
    }
  }
}

#include <algorithm>
#include "sink_container/sink_container_impl.h"

void sink_container_impl::add_sink(std::shared_ptr<media_sink> sink)
{
  if (!sink)
  {
    return;
  }
  std::scoped_lock lock(m_sink_mutex);

  for (auto it = m_sinks.begin(); it != m_sinks.end(); )
  {
    if (auto s = it->lock())
    {
      if (s == sink)
      {
        return; // already present
      }
      ++it;
    }
    else
    {
      it = m_sinks.erase(it); // cleanup
    }
  }

  m_sinks.push_back(sink);
}

void sink_container_impl::remove_sink(std::shared_ptr<media_sink> sink)
{
  std::scoped_lock lock(m_sink_mutex);
  m_sinks.erase(
    std::remove_if(m_sinks.begin(), m_sinks.end(),
      [&](const std::weak_ptr<media_sink>& w)
      {
        auto s = w.lock();
        return !s || s == sink;
      }),
    m_sinks.end());
}

std::vector<std::shared_ptr<media_sink>> sink_container_impl::snapshot_sinks()
{
  std::vector<std::shared_ptr<media_sink>> out;
  std::scoped_lock lock(m_sink_mutex);
  for (auto it = m_sinks.begin(); it != m_sinks.end(); )
  {
    if (auto s = it->lock())
    {
      out.push_back(s);
      ++it;
    }
    else
    {
      it = m_sinks.erase(it);
    }
  }
  return out;
}

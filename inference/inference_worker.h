#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "frame_sampler.h" // decoded_frame

// Stage 3 of the analytics pipeline: object detection on decoded frames, on its own thread,
// decoupled from decoding (stage 2, frame_sampler) and RTSP reception (stage 1, rtsp_reader).
// Detection (OpenVINO) is by far the slowest stage, so its input queue is bounded and drop-oldest:
// when detection falls behind, older decoded frames are discarded (freshest-frame-wins) instead of
// stalling the decoder or the reader. Stalling upstream would stop draining the RTSP socket, which
// back-pressures MediaMTX and makes the live WebRTC view stutter/fast-forward.
class inference_worker {
public:
  explicit inference_worker(std::function<void(const decoded_frame&)> detect);
  ~inference_worker();

  // Called from the decode thread; enqueues (drop-oldest) and returns immediately.
  void submit(const decoded_frame& frame);

private:
  void worker_loop();

  std::function<void(const decoded_frame&)> m_detect;

  static constexpr size_t k_max_queue = 2;
  std::mutex m_mutex;
  std::condition_variable m_cv;
  std::deque<decoded_frame> m_queue;
  std::atomic_bool m_running{true};
  std::thread m_worker;
};

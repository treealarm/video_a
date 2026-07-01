#include "face_detector.h"
#include "frame_sampler.h"
#include "logging.h"

face_detector::face_detector(const std::string& model_dir)
{
  log()->warn("face_detector: stub mode, no model loaded (model_dir='{}')", model_dir);
}

std::vector<raw_detection> face_detector::infer(const decoded_frame& /*person_crop*/)
{
  return {};
}

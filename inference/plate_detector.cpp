#include "plate_detector.h"
#include "frame_sampler.h"
#include "logging.h"

plate_detector::plate_detector(const std::string& model_dir)
{
  log()->warn("plate_detector: stub mode, no model loaded (model_dir='{}')", model_dir);
}

std::vector<raw_detection> plate_detector::infer(const decoded_frame& /*vehicle_crop*/)
{
  return {};
}

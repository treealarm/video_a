#include "plate_ocr.h"
#include "frame_sampler.h"
#include "logging.h"

plate_ocr::plate_ocr(const std::string& model_dir)
{
  log()->warn("plate_ocr: stub mode, no model loaded (model_dir='{}')", model_dir);
}

std::optional<ocr_result> plate_ocr::infer(const decoded_frame& /*plate_crop*/)
{
  return std::nullopt;
}

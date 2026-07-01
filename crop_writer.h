#pragma once

#include <cstdint>
#include <string>

struct decoded_frame;

// Writes JPEG crops under ANALYTICS_CROP_STORAGE_PATH — video_a's own storage, not shared with
// vms_rec (see project plan A.6). crop_ref returned here is opaque to the caller: VmsAnalytics
// only stores it as a string in Phase 1, it does not resolve or serve the file (no UI yet).
class crop_writer {
public:
  explicit crop_writer(std::string base_path);

  // Returns a path relative to base_path, or an empty string on failure — a failed crop write
  // must never abort the detection itself, it's best-effort.
  std::string write_crop(const std::string& watch_id, int64_t track_id, const decoded_frame& crop) const;

private:
  std::string m_base_path;
};

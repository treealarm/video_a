#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "grpc_layer/analytics_service_impl.h"
#include "grpc_layer/detection_queue.h"
#include "logging.h"
#include "roi/roi_file_job.h"
#include "watch_manager.h"

namespace {
std::string require_env(const char* name)
{
  const char* v = std::getenv(name);
  if (!v || std::string(v).empty())
  {
    log()->critical("Required environment variable '{}' is not set", name);
    std::exit(1);
  }
  return v;
}

// std::stoi would throw on a non-numeric value, aborting with no log line at all — the opposite of
// the critical+exit(1) contract every other startup check here follows. The range check matters
// just as much: a 0 or negative interval makes the pipeline's "embedding due" test always true,
// silently turning a once-per-interval ReID inference into one per person per sampled frame.
int require_env_int(const char* name, int min_value, int max_value)
{
  const std::string raw = require_env(name);
  int value = 0;
  const auto* const last = raw.data() + raw.size();
  const auto [parse_end, ec] = std::from_chars(raw.data(), last, value);
  if (ec != std::errc{} || parse_end != last)
  {
    log()->critical("Environment variable '{}' must be an integer, got '{}'", name, raw);
    std::exit(1);
  }
  if (value < min_value || value > max_value)
  {
    log()->critical("Environment variable '{}' must be between {} and {}, got {}",
      name, min_value, max_value, value);
    std::exit(1);
  }
  return value;
}

// ---------------------------------------------------------------------------
// One-shot ROI pass over a file
//
// A second entry point rather than a second binary: it needs the same models, the same pipeline
// and the same decoder, and the only thing it does differently is where the frames come from and
// where they go. Selected by --roi-file; without it nothing below runs and the process is the
// server it has always been.
// ---------------------------------------------------------------------------

void print_roi_usage()
{
  std::fprintf(stderr,
    "usage: analytics-worker --roi-file INPUT --roi-grpc HOST:PORT [options]\n"
    "\n"
    "  Decodes INPUT, detects what is in it, and re-encodes it through roi-transcode-svc\n"
    "  spending the bits on the detected objects. Writes the video and a sidecar of the\n"
    "  regions that were asked for, which roi_transcode's integration/scripts/boxes-to-ass.py\n"
    "  turns into subtitles any player can draw over the result.\n"
    "\n"
    "  --out PATH             output video (default: INPUT with .roi.mp4)\n"
    "  --boxes PATH           regions sidecar (default: INPUT with .boxes.json; '-' skips)\n"
    "  --detect-fps N         inference passes per second of content, 0 = every frame (2)\n"
    "  --classes LIST         person,face,vehicle,license_plate (all of them)\n"
    "  --min-confidence F     detector threshold (0.5)\n"
    "  --encoder NAME         auto | qsv | vaapi | nvenc | cpu (auto)\n"
    "  --codec NAME           h264 | h265 | same (same as the source)\n"
    "  --preset NAME          balanced | fastest | fast | slow | slowest (balanced)\n"
    "  --crf N                constant quantizer (encoder's own default)\n"
    "  --gop N                keyframe interval in frames (same as the source)\n"
    "  --background-qp N      how much worse the background gets, e.g. 6\n"
    "  --qp KIND=DELTA        per-kind override; omit to let §11 importance decide\n"
    "  --qp-at-zero N         importance→QP line at I=0 (default 5; product 3)\n"
    "  --qp-at-one N          importance→QP line at I=1 (default -10; product -12)\n"
    "  --pad F                halo around each box as a fraction of the frame, e.g. 0.02\n"
    "  --target-bitrate N     §15 average bitrate ceiling in bits/sec (0 = CQP/CRF only)\n"
    "  --bitrate-overshoot F  allowed average overshoot fraction, e.g. 0.1\n"
    "  --max-regions N        upper bound on regions handed to the encoder\n"
    "  --regions boxes|mask   how the regions travel; a mask costs no region budget, which\n"
    "                         matters on a driver that accepts only a handful (boxes)\n"
    "  --mask-scale N         mask is 1/N of the frame; finer than the 32px block grid is\n"
    "                         bytes on the wire for nothing (8)\n");
}

bool parse_classes(const std::string& list, std::vector<detection_kind>& out)
{
  size_t start = 0;
  while (start <= list.size())
  {
    const size_t comma = list.find(',', start);
    const std::string name = list.substr(start,
      comma == std::string::npos ? std::string::npos : comma - start);
    if (!name.empty())
    {
      if (name == "person")             out.push_back(detection_kind::person);
      else if (name == "face")          out.push_back(detection_kind::face);
      else if (name == "vehicle")       out.push_back(detection_kind::vehicle);
      else if (name == "license_plate" || name == "plate")
                                        out.push_back(detection_kind::license_plate);
      else
      {
        std::fprintf(stderr, "unknown detection class '%s'\n", name.c_str());
        return false;
      }
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return !out.empty();
}

std::string default_sibling(const std::string& input, const char* suffix)
{
  std::filesystem::path p(input);
  p.replace_extension();
  return p.string() + suffix;
}

std::string absolute_path(const std::string& raw)
{
  std::error_code ec;
  const auto canon = std::filesystem::weakly_canonical(std::filesystem::path(raw), ec);
  return ec ? std::filesystem::absolute(raw).string() : canon.string();
}

int run_roi_cli(const std::vector<std::string>& args)
{
  roi_job_config cfg;
  std::string out_path;
  std::string boxes_path;
  double pad = 0;
  std::vector<std::pair<std::string, int32_t>> qp_overrides;

  const auto need_value = [&](size_t& i) -> const std::string* {
    if (i + 1 >= args.size())
    {
      std::fprintf(stderr, "%s needs a value\n", args[i].c_str());
      return nullptr;
    }
    return &args[++i];
  };

  for (size_t i = 0; i < args.size(); ++i)
  {
    const std::string& a = args[i];
    const std::string* v = nullptr;

    if (a == "--roi-file")            { if (!(v = need_value(i))) return 2; cfg.input_path = *v; }
    else if (a == "--roi-grpc")       { if (!(v = need_value(i))) return 2; cfg.encode.target = *v; }
    else if (a == "--out")            { if (!(v = need_value(i))) return 2; out_path = *v; }
    else if (a == "--boxes")          { if (!(v = need_value(i))) return 2; boxes_path = *v; }
    else if (a == "--detect-fps")     { if (!(v = need_value(i))) return 2; cfg.detect_fps = std::stod(*v); }
    else if (a == "--min-confidence") { if (!(v = need_value(i))) return 2; cfg.min_confidence = std::stof(*v); }
    else if (a == "--encoder")        { if (!(v = need_value(i))) return 2; cfg.encode.encoder = *v; }
    else if (a == "--codec")          { if (!(v = need_value(i))) return 2; cfg.encode.codec = *v; }
    else if (a == "--preset")         { if (!(v = need_value(i))) return 2; cfg.encode.preset = *v; }
    else if (a == "--crf")            { if (!(v = need_value(i))) return 2; cfg.encode.crf = std::stoi(*v); }
    else if (a == "--gop")            { if (!(v = need_value(i))) return 2; cfg.encode.gop = std::stoi(*v); }
    else if (a == "--max-regions")    { if (!(v = need_value(i))) return 2; cfg.encode.max_regions = std::stoi(*v); }
    else if (a == "--pad")            { if (!(v = need_value(i))) return 2; pad = std::stod(*v); }
    else if (a == "--mask-scale")     { if (!(v = need_value(i))) return 2; cfg.mask_scale = std::stoi(*v); }
    else if (a == "--regions")
    {
      if (!(v = need_value(i))) return 2;
      if (*v == "boxes")     cfg.regions = roi_region_form::boxes;
      else if (*v == "mask") cfg.regions = roi_region_form::mask;
      else
      {
        std::fprintf(stderr, "--regions wants boxes or mask, got '%s'\n", v->c_str());
        return 2;
      }
    }
    else if (a == "--background-qp")
    {
      if (!(v = need_value(i))) return 2;
      cfg.encode.background_qp_delta = std::stoi(*v);
      cfg.encode.has_background_qp_delta = true;
    }
    else if (a == "--classes")
    {
      if (!(v = need_value(i))) return 2;
      if (!parse_classes(*v, cfg.classes)) return 2;
    }
    else if (a == "--qp")
    {
      if (!(v = need_value(i))) return 2;
      const auto eq = v->find('=');
      if (eq == std::string::npos)
      {
        std::fprintf(stderr, "--qp wants KIND=DELTA, got '%s'\n", v->c_str());
        return 2;
      }
      qp_overrides.emplace_back(v->substr(0, eq), std::stoi(v->substr(eq + 1)));
    }
    else if (a == "--qp-at-zero")
    {
      if (!(v = need_value(i))) return 2;
      cfg.encode.qp_at_zero = std::stoi(*v);
      cfg.encode.has_qp_at_zero = true;
    }
    else if (a == "--qp-at-one")
    {
      if (!(v = need_value(i))) return 2;
      cfg.encode.qp_at_one = std::stoi(*v);
      cfg.encode.has_qp_at_one = true;
    }
    else if (a == "--target-bitrate")
    {
      if (!(v = need_value(i))) return 2;
      cfg.encode.target_bitrate = static_cast<uint64_t>(std::stoull(*v));
    }
    else if (a == "--bitrate-overshoot")
    {
      if (!(v = need_value(i))) return 2;
      cfg.encode.max_average_overshoot = std::stof(*v);
    }
    else if (a == "--help" || a == "-h") { print_roi_usage(); return 0; }
    else
    {
      std::fprintf(stderr, "unknown option '%s'\n", a.c_str());
      print_roi_usage();
      return 2;
    }
  }

  if (cfg.input_path.empty() || cfg.encode.target.empty())
  {
    print_roi_usage();
    return 2;
  }
  if (cfg.classes.empty())
  {
    cfg.classes = { detection_kind::person, detection_kind::face,
      detection_kind::vehicle, detection_kind::license_plate };
  }

  cfg.input_path = absolute_path(cfg.input_path);
  cfg.encode.output_path = absolute_path(
    out_path.empty() ? default_sibling(cfg.input_path, ".roi.mp4") : out_path);
  if (boxes_path != "-")
  {
    cfg.boxes_json_path = absolute_path(
      boxes_path.empty() ? default_sibling(cfg.input_path, ".boxes.json") : boxes_path);
  }

  for (auto& [kind, delta] : qp_overrides)
    cfg.encode.kinds.push_back(roi_kind_quality{ kind, delta, pad });
  // Without per-kind rows the pad rides on RoiQuality.pad -- that is the importance path,
  // and it is what the archive pass uses. With per-kind rows each kind carries its own.
  if (pad > 0)
    cfg.encode.pad = pad;

  // Only what this mode actually uses. ANALYTICS_GRPC_PORT is deliberately not required: nothing
  // here listens, and demanding it would make an offline pass depend on a port it never binds.
  cfg.model_dir = require_env("ANALYTICS_MODEL_PATH");
  require_env("ANALYTICS_DEVICE");
  cfg.reid_embed_interval_sec = require_env_int("ANALYTICS_REID_EMBED_INTERVAL_SEC", 1, 86400);

  return run_roi_file_job(cfg) ? 0 : 1;
}
}

int main(int argc, char** argv)
{
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const auto& a : args)
  {
    if (a == "--roi-file" || a == "--help" || a == "-h")
      return run_roi_cli(args);
  }

  const auto grpc_port = require_env("ANALYTICS_GRPC_PORT");
  const auto model_dir = require_env("ANALYTICS_MODEL_PATH");
  // Reserved for future OpenVINO device selection (CPU/GPU) — validated now so a missing value
  // is visible immediately, even though stub inference doesn't use it yet.
  require_env("ANALYTICS_DEVICE");
  // How often (seconds) a still-live track recomputes its re-id embedding. The vector itself rides
  // along with every detection of the track; this only bounds the inference.
  const int reid_embed_interval_sec = require_env_int("ANALYTICS_REID_EMBED_INTERVAL_SEC", 1, 86400);

  auto queue = std::make_shared<detection_queue>();

  auto watches = std::make_shared<watch_manager>(
    model_dir,
    reid_embed_interval_sec,
    [queue](const std::string& watch_id, const final_detection& det)
    {
      queue->push(queued_detection{ watch_id, det });
    });

  analytics_service_impl service(watches, queue);

  const std::string server_address = "0.0.0.0:" + grpc_port;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  if (!server)
  {
    log()->critical("Failed to start gRPC server on {}", server_address);
    return 1;
  }

  log()->info("analytics-worker listening on {}", server_address);
  server->Wait();
  return 0;
}

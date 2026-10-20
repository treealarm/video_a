#pragma once

#include <atomic>
#include <string>

#include <grpcpp/grpcpp.h>

#include "frame_analysis.grpc.pb.h"

// Serves frame_analysis.v1.FrameAnalysis: one offline pass over a file per call, streaming every
// decoded frame with the objects in it. The same decode-and-detect loop as --roi-file (see
// roi/file_analyzer.h), with the frames going to the caller instead of to an encoder.
//
// Each call builds its own pipeline -- models, tracker and all -- because a pass is a clip of its
// own and must not inherit track ids or motion history from whatever ran before it. That is also
// what makes a pass expensive, so how many may run at once is bounded.
class frame_analysis_service_impl final : public frame_analysis::v1::FrameAnalysis::Service {
public:
  frame_analysis_service_impl(std::string model_dir, int reid_embed_interval_sec, int max_jobs);

  grpc::Status AnalyzeFile(
    grpc::ServerContext* context,
    const frame_analysis::v1::AnalyzeFileRequest* request,
    grpc::ServerWriter<frame_analysis::v1::AnalyzeFileEvent>* writer) override;

private:
  std::string m_model_dir;
  int m_reid_embed_interval_sec = 15;
  int m_max_jobs = 1;
  std::atomic_int m_running{ 0 };
};

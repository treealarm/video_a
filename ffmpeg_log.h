#pragma once

// Routes libav*'s own av_log output into our spdlog logger, collapsing bursts of an identical
// message down to one line every few seconds.
//
// The motivating case is a camera that drops its TCP connection: av_read_frame then fails
// immediately and forever until the reader's watchdog reconnects, and every one of those failures
// prints "Failed reading RTSP data: End of file" straight to stderr — thousands of lines for one
// dead stream. FFmpeg's own AV_LOG_SKIP_REPEATED cannot help: it only collapses lines that repeat
// back-to-back from the same context, so two watches failing at once interleave and defeat it.
//
// Call once, before anything opens an AVFormatContext. Safe to call more than once.
void install_ffmpeg_log_bridge();

// Feed an H.264 file through FFmpeg's h264_nvmpi decoder at camera pace and report how far behind the input each
// decoded frame comes out. Reads frames the way a live consumer does: take every frame the decoder offers, and only
// read the next packet once it has none. Packets "arrive" on a wall-clock schedule, so a consumer stall leaves a
// burst of them waiting, as a socket buffer would.
//
// Build (against the FFmpeg under test):
//   g++ -O2 -std=c++17 tools/dec_lag.cpp -o dec_lag -I<prefix>/include -L<prefix>/lib -lavformat -lavcodec -lavutil
// Run:
//   LD_LIBRARY_PATH=<ffmpeg prefix>/lib:<libnvmpi dir> ./dec_lag clip.mp4 [options]
// Options:
//   --fps N        packet arrival rate (default: the stream's frame rate)
//   --frames N     stop after N frames (default 600)
//   --stall-at N   after the Nth frame, stop reading for --stall-ms (default: no stall)
//   --stall-ms N   length of the stall (default 1000)
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace
{
    double milliseconds(Clock::duration duration)
    {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    double percentile(std::vector<double> values, double fraction)
    {
        if (values.empty())
        {
            return 0.0;
        }

        std::sort(values.begin(), values.end());
        return values[std::min(values.size() - 1, static_cast<std::size_t>(values.size() * fraction))];
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s clip.mp4 [--fps N] [--frames N] [--stall-at N] [--stall-ms N]\n", argv[0]);
        return 2;
    }

    double fps = 0.0;
    int frameLimit = 600;
    int stallAt = -1;
    int stallMilliseconds = 1000;
    for (int i = 2; i + 1 < argc; i += 2)
    {
        std::string option = argv[i];
        double value = std::atof(argv[i + 1]);
        if (option == "--fps")
        {
            fps = value;
        }
        else if (option == "--frames")
        {
            frameLimit = static_cast<int>(value);
        }
        else if (option == "--stall-at")
        {
            stallAt = static_cast<int>(value);
        }
        else if (option == "--stall-ms")
        {
            stallMilliseconds = static_cast<int>(value);
        }
    }

    AVFormatContext* format = nullptr;
    if (avformat_open_input(&format, argv[1], nullptr, nullptr) < 0 || avformat_find_stream_info(format, nullptr) < 0)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

    int videoIndex = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    AVStream* stream = format->streams[videoIndex];
    if (fps <= 0.0)
    {
        fps = av_q2d(stream->avg_frame_rate);
    }

    const AVCodec* codec = avcodec_find_decoder_by_name("h264_nvmpi");
    AVCodecContext* decoder = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(decoder, stream->codecpar);
    if (!codec || avcodec_open2(decoder, codec, nullptr) < 0)
    {
        std::fprintf(stderr, "cannot open h264_nvmpi\n");
        return 1;
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::map<int64_t, std::pair<Clock::time_point, int>> arrivals;    // pts -> when its packet "arrived", its index
    std::vector<double> lagMilliseconds;
    std::vector<int> lagFrames;
    const auto interval = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));
    const auto start = Clock::now();
    int packetsRead = 0;
    int framesOut = 0;
    bool endOfFile = false;
    while (framesOut < frameLimit)
    {
        int result = avcodec_receive_frame(decoder, frame);
        if (result == 0)
        {
            auto now = Clock::now();
            auto arrival = arrivals.find(frame->pts);
            if (arrival != arrivals.end())
            {
                lagMilliseconds.push_back(milliseconds(now - arrival->second.first));
                lagFrames.push_back(packetsRead - 1 - arrival->second.second);
                arrivals.erase(arrivals.begin(), std::next(arrival));
            }

            ++framesOut;
            av_frame_unref(frame);
            if (framesOut == stallAt)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(stallMilliseconds));
            }

            if (framesOut % 60 == 0 && !lagFrames.empty())
            {
                std::printf("frame %4d: %2d frames behind the newest packet, %6.1f ms after its own arrived\n",
                    framesOut, lagFrames.back(), lagMilliseconds.back());
            }

            continue;
        }

        if (result == AVERROR_EOF || (result != AVERROR(EAGAIN)))
        {
            break;
        }

        if (endOfFile)
        {
            continue;
        }

        // Wait for the next packet to arrive, as a live socket would deliver it
        auto due = start + interval * packetsRead;
        std::this_thread::sleep_until(due);
        if (av_read_frame(format, packet) < 0)
        {
            endOfFile = true;
            avcodec_send_packet(decoder, nullptr);
            continue;
        }

        if (packet->stream_index == videoIndex)
        {
            arrivals[packet->pts] = {due, packetsRead};
            ++packetsRead;
            avcodec_send_packet(decoder, packet);
        }

        av_packet_unref(packet);
    }

    std::map<int, int> histogram;
    for (int lag : lagFrames)
    {
        ++histogram[lag];
    }

    std::printf("frames behind -> count:");
    for (const auto& [lag, count] : histogram)
    {
        std::printf(" %d->%d", lag, count);
    }

    std::printf("\n");
    std::vector<double> behind(lagFrames.begin(), lagFrames.end());
    std::vector<double> tail(behind.end() - std::min<std::ptrdiff_t>(behind.size(), 120), behind.end());
    std::vector<double> tailMilliseconds(
        lagMilliseconds.end() - std::min<std::ptrdiff_t>(lagMilliseconds.size(), 120), lagMilliseconds.end());
    std::printf("%d frames at %.1f fps. Frames behind the newest packet: median %.0f, p90 %.0f, max %.0f; "
                "last 120 frames: median %.0f (%.0f ms after arrival)\n",
        framesOut, fps, percentile(behind, 0.5), percentile(behind, 0.9), percentile(behind, 1.0),
        percentile(tail, 0.5), percentile(tailMilliseconds, 0.5));

    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    avformat_close_input(&format);
    return 0;
}

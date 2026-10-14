// Decode an H.264 mp4 through nvmpi while other threads keep the GPU busy on blocking CUDA streams, the way
// TensorRT inference does in sendisapp. Reports how long nvmpi_decoder_get_frame takes and checks every decoded
// frame against libavcodec's software decode of the same packet (matched by pts). The hardware decoder is not
// bit-exact with libavcodec, so a frame only counts as corrupt when a visible share of its luma differs grossly.
// It can also imitate sendisapp's consumer: an asynchronous read of each frame's planes on a blocking stream, to
// catch frame buffers being reused while a read is still pending.
//
// Build (on the Jetson, against the libnvmpi under test):
//   nvcc -O2 -std=c++17 -I<jetson-ffmpeg>/include tools/dec_gpu_contention.cu -o dec_gpu_contention \
//       -L<libnvmpi dir> -lnvmpi -lavformat -lavcodec -lavutil
// Run:
//   LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:<libnvmpi dir> ./dec_gpu_contention clip.mp4 [options]
// Options:
//   --load-threads N    threads keeping a blocking stream busy (default 2)
//   --load-ms N         length of each queued busy kernel in ms (default 8)
//   --consumer MODE     none (default) | async | sync: read each frame's planes on a blocking stream
//   --consumer-delay N  ms of busy work queued ahead of each consumer read (default 0)
//   --frames N          stop after N frames
//   --fps N             feed packets at N per second, like a live camera (default: as fast as possible)
#include <nvmpi.h>
#include <cuda_runtime.h>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#define CHECK(call)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = (call);                                                                                   \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(status));            \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

__global__ void spin(long long cycles)
{
    long long start = clock64();
    while (clock64() - start < cycles)
    {
    }
}

static long long cyclesPerMs()
{
    int clockKHz = 0;
    CHECK(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0));
    return static_cast<long long>(clockKHz);
}

using Luma = std::vector<unsigned char>;

// Pairs hardware and software luma planes by pts and counts frames that differ grossly
struct Checker
{
    std::map<int64_t, Luma> hardware, software;
    size_t compared{0}, corrupt{0};
    double worstShare{0};

    void addHardware(int64_t pts, Luma luma)
    {
        hardware[pts] = std::move(luma);
        match(pts);
    }

    void addSoftware(int64_t pts, Luma luma)
    {
        software[pts] = std::move(luma);
        match(pts);
    }

    void match(int64_t pts)
    {
        auto h = hardware.find(pts), s = software.find(pts);
        if (h == hardware.end() || s == software.end())
        {
            return;
        }

        size_t gross = 0;
        for (size_t i = 0; i < h->second.size(); ++i)
        {
            gross += std::abs(int(h->second[i]) - int(s->second[i])) > 32;
        }

        double share = double(gross) / h->second.size();
        worstShare = std::max(worstShare, share);
        corrupt += share > 0.001;
        ++compared;
        hardware.erase(h);
        software.erase(s);
    }
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s clip.mp4 [options]\n", argv[0]);
        return 2;
    }

    int loadThreads = 2, loadMs = 8, consumerDelayMs = 0, maxFrames = 1 << 30, feedFps = 0;
    std::string consumer = "none";
    for (int i = 2; i + 1 < argc; i += 2)
    {
        std::string option = argv[i];
        if (option == "--load-threads")
            loadThreads = std::atoi(argv[i + 1]);
        else if (option == "--load-ms")
            loadMs = std::atoi(argv[i + 1]);
        else if (option == "--consumer")
            consumer = argv[i + 1];
        else if (option == "--consumer-delay")
            consumerDelayMs = std::atoi(argv[i + 1]);
        else if (option == "--frames")
            maxFrames = std::atoi(argv[i + 1]);
        else if (option == "--fps")
            feedFps = std::atoi(argv[i + 1]);
    }

    long long msCycles = cyclesPerMs();
    std::atomic<bool> stop{false};
    std::vector<std::thread> loaders;
    for (int i = 0; i < loadThreads; ++i)
    {
        loaders.emplace_back(
            [&]()
            {
                cudaStream_t stream;
                CHECK(cudaStreamCreate(&stream));    // blocking, like TensorRT engine streams in sendisapp
                while (!stop.load())
                {
                    spin<<<1, 1, 0, stream>>>(loadMs * msCycles);
                    spin<<<1, 1, 0, stream>>>(loadMs * msCycles);
                    CHECK(cudaStreamSynchronize(stream));
                }

                CHECK(cudaStreamDestroy(stream));
            });
    }

    AVFormatContext* format = nullptr;
    if (avformat_open_input(&format, argv[1], nullptr, nullptr) < 0 || avformat_find_stream_info(format, nullptr) < 0)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

    int videoIndex = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const AVCodecParameters* parameters = format->streams[videoIndex]->codecpar;
    const size_t width = parameters->width, height = parameters->height;

    AVBSFContext* bsf = nullptr;
    av_bsf_alloc(av_bsf_get_by_name("h264_mp4toannexb"), &bsf);
    avcodec_parameters_copy(bsf->par_in, parameters);
    av_bsf_init(bsf);

    const AVCodec* h264 = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext* softwareDecoder = avcodec_alloc_context3(h264);
    avcodec_parameters_to_context(softwareDecoder, parameters);
    softwareDecoder->thread_count = 4;
    avcodec_open2(softwareDecoder, h264, nullptr);
    AVFrame* softwareFrame = av_frame_alloc();

    nvDecParam param{};
    param.codingType = NV_VIDEO_CodingH264;
    param.pixFormat = NV_PIX_YUV420;
    param.frame_pool_size = 12;
    nvmpictx* decoder = nvmpi_create_decoder(&param);
    if (!decoder)
    {
        std::fprintf(stderr, "nvmpi_create_decoder failed\n");
        return 1;
    }

    // Blocking streams, like sendisapp's: the stock decoder's plane copy is an asynchronous device-to-device copy on
    // the legacy default stream, which only blocking streams wait for
    cudaStream_t checkStream, consumerStream;
    CHECK(cudaStreamCreate(&checkStream));
    CHECK(cudaStreamCreate(&consumerStream));

    Checker checker;
    std::vector<double> getFrameMs;
    std::vector<std::pair<int64_t, unsigned char*>> consumed;    // luma copies made by the consumer

    auto handle = [&](const nvFrame& frame)
    {
        int64_t pts = static_cast<int64_t>(frame.timestamp);
        if (consumer != "none")
        {
            unsigned char* copy = nullptr;
            CHECK(cudaMalloc(&copy, width * height));
            if (consumerDelayMs > 0)
            {
                spin<<<1, 1, 0, consumerStream>>>(consumerDelayMs * msCycles);
            }

            CHECK(cudaMemcpy2DAsync(copy, width, frame.payload[0], frame.linesize[0], width, height,
                cudaMemcpyDeviceToDevice, consumerStream));
            consumed.emplace_back(pts, copy);
            if (consumer == "sync")
            {
                CHECK(cudaStreamSynchronize(consumerStream));
            }

            return;
        }

        Luma luma(width * height);
        CHECK(cudaMemcpy2DAsync(luma.data(), width, frame.payload[0], frame.linesize[0], width, height,
            cudaMemcpyDeviceToHost, checkStream));
        CHECK(cudaStreamSynchronize(checkStream));
        checker.addHardware(pts, std::move(luma));
    };

    auto drain = [&]()
    {
        while (static_cast<int>(getFrameMs.size()) < maxFrames)
        {
            nvFrame frame{};
            auto start = std::chrono::steady_clock::now();
            if (nvmpi_decoder_get_frame(decoder, &frame, false) < 0)
            {
                return;
            }

            getFrameMs.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            handle(frame);
        }
    };

    auto receiveSoftware = [&]()
    {
        while (avcodec_receive_frame(softwareDecoder, softwareFrame) == 0)
        {
            Luma luma(width * height);
            for (size_t y = 0; y < height; ++y)
            {
                std::memcpy(luma.data() + y * width, softwareFrame->data[0] + y * softwareFrame->linesize[0], width);
            }

            checker.addSoftware(softwareFrame->pts, std::move(luma));
        }
    };

    AVPacket* packet = av_packet_alloc();
    AVPacket* softwarePacket = av_packet_alloc();
    auto wallStart = std::chrono::steady_clock::now();
    size_t fed = 0;
    while (av_read_frame(format, packet) >= 0 && static_cast<int>(getFrameMs.size()) < maxFrames)
    {
        if (packet->stream_index == videoIndex)
        {
            if (feedFps > 0)
            {
                auto due = wallStart + std::chrono::microseconds(1000000 * fed / feedFps);
                while (std::chrono::steady_clock::now() < due)
                {
                    drain();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            ++fed;
            av_packet_ref(softwarePacket, packet);
            avcodec_send_packet(softwareDecoder, softwarePacket);
            av_packet_unref(softwarePacket);
            receiveSoftware();

            if (av_bsf_send_packet(bsf, packet) == 0)
            {
                while (av_bsf_receive_packet(bsf, packet) == 0)
                {
                    nvPacket input{};
                    input.payload = packet->data;
                    input.payload_size = packet->size;
                    input.pts = packet->pts;
                    nvmpi_decoder_put_packet(decoder, &input);
                    av_packet_unref(packet);
                    drain();
                }
            }
        }

        av_packet_unref(packet);
    }

    avcodec_send_packet(softwareDecoder, nullptr);
    receiveSoftware();

    // Flush: wait for the hardware decoder to hand back what it still holds
    nvPacket end{};
    nvmpi_decoder_put_packet(decoder, &end);
    for (auto idle = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - idle < std::chrono::seconds(2);)
    {
        size_t before = getFrameMs.size();
        drain();
        if (getFrameMs.size() != before)
        {
            idle = std::chrono::steady_clock::now();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    if (consumer != "none")
    {
        CHECK(cudaStreamSynchronize(consumerStream));
        for (auto& [pts, copy] : consumed)
        {
            Luma luma(width * height);
            CHECK(cudaMemcpy(luma.data(), copy, luma.size(), cudaMemcpyDeviceToHost));
            checker.addHardware(pts, std::move(luma));
            CHECK(cudaFree(copy));
        }
    }

    stop = true;
    for (std::thread& loader : loaders)
    {
        loader.join();
    }

    nvmpi_decoder_close(decoder);

    std::vector<double> sorted = getFrameMs;
    std::sort(sorted.begin(), sorted.end());
    auto at = [&](double q)
    { return sorted.empty() ? 0.0 : sorted[std::min(sorted.size() - 1, size_t(q * sorted.size()))]; };
    double total = 0;
    for (double ms : getFrameMs)
    {
        total += ms;
    }

    std::printf("frames %zu in %.1f s (%.1f fps); get_frame ms median %.2f p90 %.2f p99 %.2f max %.2f, total %.1f s; "
                "load %d x %d ms; consumer %s delay %d ms; corrupt %zu of %zu compared (worst %.2f%% of luma), "
                "unmatched hw %zu\n",
        getFrameMs.size(), wallSeconds, getFrameMs.size() / wallSeconds, at(0.5), at(0.9), at(0.99), at(1.0),
        total / 1000, loadThreads, loadMs, consumer.c_str(), consumerDelayMs, checker.corrupt, checker.compared,
        checker.worstShare * 100, checker.hardware.size());
    return checker.corrupt ? 1 : 0;
}

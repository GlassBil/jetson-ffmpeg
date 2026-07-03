// On-device tear-repro harness for the WITH_CUDA_BUFFERS encoder input path: feed solid-color
// YUV420 frames (a new, strongly distinct color every frame) through the nvmpi h264 encoder as
// fast as it accepts them, and write the resulting Annex-B stream to a file. Any input-copy race
// (encoder reading the EGL-mapped surface before the CUDA copy has landed) shows up in the decoded
// output as regions of the *previous* frame's color or missing chroma, which
// tools/enc_tear_check.py detects. Optionally spawns a CUDA copy-engine load thread to mimic the
// GPU contention sendisapp's inference produces. Build with -DBUILD_ENC_TEAR_SOAK=ON; needs the
// Jetson hardware encoder. First-run null create => "no hardware encoder", exit 0.
//
// Usage: enc_tear_soak [frames] [outfile] [load 0|1]   (default 300 out.h264 0)

#include <nvmpi.h>

#ifdef WITH_CUDA_BUFFERS
#include <cuda_runtime_api.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>
#include <vector>

namespace
{
	const int WIDTH = 1280;
	const int HEIGHT = 800;
	const int POOL_SIZE = 8;

	struct YuvColor
	{
		uint8_t y;
		uint8_t u;
		uint8_t v;
	};

	// Four colors far apart in every plane so a torn region of the previous frame's pixels (or
	// zeroed chroma) always deviates strongly from the expected solid color.
	// Must match COLORS in tools/enc_tear_check.py.
	const YuvColor COLORS[4] = {
		{81, 90, 240},     // red
		{145, 54, 34},     // green
		{41, 240, 110},    // blue
		{210, 16, 146},    // yellow
	};

	void fillParam(nvEncParam& param)
	{
		memset(&param, 0, sizeof(param));
		param.width = WIDTH;
		param.height = HEIGHT;
		param.bitrate = 4000000;
		param.profile = 77;    // FF_PROFILE_H264_MAIN
		param.level = 51;
		param.capture_num = 10;
		param.fps_n = 25;
		param.fps_d = 1;
		param.hw_preset_type = 3;
		param.codingType = NV_VIDEO_CodingH264;
		param.insert_spspps_idr = 1;
		param.qmin = 0;    // 0 => encoder skips setQpRange (qmin/qmax are unsigned)
		param.qmax = 0;
		param.max_b_frames = 0;
		param.refs = 0;
	}

	// Write every filled packet to the output file and recycle it into the empty pool.
	// Returns the number of packets written, or stops (returning what it wrote so far) once the
	// encoder reports EOS.
	int drainFilled(nvmpictx* ctx, FILE* out, bool untilEos)
	{
		int written = 0;
		for(;;)
		{
			nvPacket* packet = nullptr;
			int ret = nvmpi_encoder_get_packet(ctx, &packet);
			if(ret != 0)
			{
				if(ret == -2 || !untilEos)
				{
					break;    // -2 = EOS observed; -1 = nothing pending right now
				}

				continue;
			}

			fwrite(packet->payload, 1, packet->payload_size, out);
			written++;
			nvmpi_encoder_qEmptyPacket(ctx, packet);
		}

		return written;
	}

#ifdef WITH_CUDA_BUFFERS
	// Keep the copy engines busy with large device-to-device copies on non-blocking streams, the
	// same contention profile heavy inference puts next to the encoder in sendisapp.
	void cudaLoadThread(std::atomic<bool>* stop)
	{
		const size_t LOAD_BYTES = 128u * 1024 * 1024;
		void* src = nullptr;
		void* dst = nullptr;
		cudaStream_t streams[2] = {};
		if(cudaMalloc(&src, LOAD_BYTES) != cudaSuccess || cudaMalloc(&dst, LOAD_BYTES) != cudaSuccess)
		{
			fprintf(stderr, "load thread: cudaMalloc failed, running without GPU load\n");
			return;
		}

		for(cudaStream_t& stream : streams)
		{
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		}

		while(!stop->load())
		{
			for(cudaStream_t& stream : streams)
			{
				cudaMemcpyAsync(dst, src, LOAD_BYTES, cudaMemcpyDeviceToDevice, stream);
			}

			for(cudaStream_t& stream : streams)
			{
				cudaStreamSynchronize(stream);
			}
		}

		for(cudaStream_t& stream : streams)
		{
			cudaStreamDestroy(stream);
		}

		cudaFree(src);
		cudaFree(dst);
	}
#endif
}

int main(int argc, char** argv)
{
	const int frames = (argc > 1) ? atoi(argv[1]) : 300;
	const char* outPath = (argc > 2) ? argv[2] : "out.h264";
	const bool withLoad = (argc > 3) ? (atoi(argv[3]) != 0) : false;

	FILE* out = fopen(outPath, "wb");
	if(!out)
	{
		fprintf(stderr, "cannot open %s for writing\n", outPath);
		return 1;
	}

	nvEncParam param;
	fillParam(param);

	nvmpictx* ctx = nvmpi_create_encoder(&param);
	if(!ctx)
	{
		printf("no hardware encoder available; skipping\n");
		fclose(out);
		return 0;
	}

	std::vector<nvPacket*> pool;
	for(int k = 0; k < POOL_SIZE; k++)
	{
		nvPacket* packet = static_cast<nvPacket*>(calloc(1, sizeof(nvPacket)));
		packet->payload = static_cast<unsigned char*>(malloc(NVMPI_ENC_CHUNK_SIZE));
		nvmpi_encoder_qEmptyPacket(ctx, packet);
		pool.push_back(packet);
	}

	std::atomic<bool> stopLoad{false};
	std::thread loadThread;
#ifdef WITH_CUDA_BUFFERS
	if(withLoad)
	{
		loadThread = std::thread(cudaLoadThread, &stopLoad);
	}
#else
	if(withLoad)
	{
		fprintf(stderr, "built without WITH_CUDA_BUFFERS; GPU load unavailable\n");
	}
#endif

	// Pageable host memory, refilled and reused every frame — the same source-buffer profile as
	// the sws_scale output the FFmpeg wrapper feeds through nvmpi_encoder_put_frame.
	std::vector<uint8_t> luma(static_cast<size_t>(WIDTH) * HEIGHT);
	std::vector<uint8_t> chromaU(static_cast<size_t>(WIDTH / 2) * (HEIGHT / 2));
	std::vector<uint8_t> chromaV(static_cast<size_t>(WIDTH / 2) * (HEIGHT / 2));

	int packetsWritten = 0;
	for(int f = 0; f < frames; f++)
	{
		const YuvColor& color = COLORS[f % 4];
		memset(luma.data(), color.y, luma.size());
		memset(chromaU.data(), color.u, chromaU.size());
		memset(chromaV.data(), color.v, chromaV.size());

		nvFrame frame;
		memset(&frame, 0, sizeof(frame));
		frame.payload[0] = luma.data();
		frame.payload[1] = chromaU.data();
		frame.payload[2] = chromaV.data();
		frame.payload_size[0] = luma.size();
		frame.payload_size[1] = chromaU.size();
		frame.payload_size[2] = chromaV.size();
		frame.linesize[0] = WIDTH;
		frame.linesize[1] = WIDTH / 2;
		frame.linesize[2] = WIDTH / 2;
		frame.type = NV_PIX_YUV420;
		frame.width = WIDTH;
		frame.height = HEIGHT;
		frame.timestamp = static_cast<time_t>(f) * 40000;

		if(nvmpi_encoder_put_frame(ctx, &frame) != 0)
		{
			fprintf(stderr, "put_frame failed at frame %d\n", f);
			break;
		}

		packetsWritten += drainFilled(ctx, out, false);
	}

	nvmpi_encoder_put_frame(ctx, nullptr);    // EOS
	packetsWritten += drainFilled(ctx, out, true);

	stopLoad.store(true);
	if(loadThread.joinable())
	{
		loadThread.join();
	}

	nvPacket* packet = nullptr;
	while(nvmpi_encoder_dqEmptyPacket(ctx, &packet) == 0)
	{
		free(packet->payload);
		free(packet);
	}

	nvmpi_encoder_close(ctx);
	fclose(out);

	printf("fed %d frames, wrote %d packets to %s (load=%d)\n", frames, packetsWritten, outPath, withLoad ? 1 : 0);
	return 0;
}

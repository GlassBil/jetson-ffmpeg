// On-device soak harness: loop nvmpi_create_encoder -> feed frames -> nvmpi_encoder_close
// many times and assert open-fd count and VmRSS stay flat, to prove the encoder create/destroy
// cycle does not leak. Build with -DBUILD_ENC_LEAK_SOAK=ON (OFF by default); needs the Jetson
// hardware encoder, so it is never built or run in CI. First-iteration null create => "no
// hardware encoder", exit 0, so a deploy script can invoke it unconditionally.
//
// Usage: enc_leak_soak [iterations]   (default 500)

#include <nvmpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <vector>
#include <dirent.h>

namespace
{
	const int WIDTH = 1280;
	const int HEIGHT = 800;
	const int FRAMES_PER_ITER = 8;
	const int POOL_SIZE = 5;    // empty packets recycled per iteration (mirrors the FFmpeg wrapper)

	long readVmRssKb()
	{
		FILE* status = fopen("/proc/self/status", "r");
		if(!status)
		{
			return -1;
		}

		char line[256];
		long kb = -1;
		while(fgets(line, sizeof(line), status))
		{
			if(strncmp(line, "VmRSS:", 6) == 0)
			{
				kb = strtol(line + 6, nullptr, 10);
				break;
			}
		}

		fclose(status);
		return kb;
	}

	int countOpenFds()
	{
		DIR* dir = opendir("/proc/self/fd");
		if(!dir)
		{
			return -1;
		}

		int count = 0;
		while(struct dirent* entry = readdir(dir))
		{
			if(entry->d_name[0] != '.')
			{
				count++;
			}
		}

		closedir(dir);
		return count;
	}

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

	// Recycle every filled packet the encoder produced back into the empty pool.
	void drainFilled(nvmpictx* ctx)
	{
		nvPacket* packet = nullptr;
		while(nvmpi_encoder_get_packet(ctx, &packet) == 0)
		{
			nvmpi_encoder_qEmptyPacket(ctx, packet);
		}
	}
}

int main(int argc, char** argv)
{
	const int iterations = (argc > 1) ? atoi(argv[1]) : 500;

	// One reused YUV420 planar source frame (content is irrelevant to a leak test).
	std::vector<uint8_t> luma(static_cast<size_t>(WIDTH) * HEIGHT, 128);
	std::vector<uint8_t> chroma(static_cast<size_t>(WIDTH / 2) * (HEIGHT / 2), 128);

	int baselineFds = -1;
	long baselineRss = -1;

	for(int iter = 0; iter < iterations; iter++)
	{
		nvEncParam param;
		fillParam(param);

		nvmpictx* ctx = nvmpi_create_encoder(&param);
		if(!ctx)
		{
			if(iter == 0)
			{
				printf("no hardware encoder available; skipping soak\n");
				return 0;
			}

			printf("FAIL: create returned null at iteration %d\n", iter);
			return 1;
		}

		// Preallocate empty packets so the capture callback has somewhere to put output (and we
		// exercise the packet-pool teardown), instead of dropping every packet.
		std::vector<nvPacket*> pool;
		for(int k = 0; k < POOL_SIZE; k++)
		{
			nvPacket* packet = static_cast<nvPacket*>(calloc(1, sizeof(nvPacket)));
			packet->payload = static_cast<unsigned char*>(malloc(NVMPI_ENC_CHUNK_SIZE));
			nvmpi_encoder_qEmptyPacket(ctx, packet);
			pool.push_back(packet);
		}

		for(int f = 0; f < FRAMES_PER_ITER; f++)
		{
			nvFrame frame;
			memset(&frame, 0, sizeof(frame));
			frame.payload[0] = luma.data();
			frame.payload[1] = chroma.data();
			frame.payload[2] = chroma.data();
			frame.payload_size[0] = luma.size();
			frame.payload_size[1] = chroma.size();
			frame.payload_size[2] = chroma.size();
			frame.linesize[0] = WIDTH;
			frame.linesize[1] = WIDTH / 2;
			frame.linesize[2] = WIDTH / 2;
			frame.type = NV_PIX_YUV420;
			frame.width = WIDTH;
			frame.height = HEIGHT;
			frame.timestamp = static_cast<time_t>(f) * 40000;

			nvmpi_encoder_put_frame(ctx, &frame);
			drainFilled(ctx);
		}

		nvmpi_encoder_put_frame(ctx, nullptr);    // EOS
		drainFilled(ctx);                          // returns once EOS is observed

		// Free the packet pool: after the EOS drain every packet is back in the empty queue.
		nvPacket* packet = nullptr;
		while(nvmpi_encoder_dqEmptyPacket(ctx, &packet) == 0)
		{
			free(packet->payload);
			free(packet);
		}

		nvmpi_encoder_close(ctx);

		// Baseline after a warm-up iteration (the first create does one-time driver allocation).
		if(iter == 1)
		{
			baselineFds = countOpenFds();
			baselineRss = readVmRssKb();
		}

		if(iter > 1 && iter % 50 == 0)
		{
			int fds = countOpenFds();
			long rss = readVmRssKb();
			printf("iter %d: fds=%d (base %d) rss=%ldKB (base %ldKB)\n", iter, fds, baselineFds, rss, baselineRss);

			if(baselineFds >= 0 && fds > baselineFds + 4)
			{
				printf("FAIL: fd leak -- %d fds vs baseline %d\n", fds, baselineFds);
				return 1;
			}

			if(baselineRss > 0 && rss > baselineRss + 65536)    // >64 MB growth = leak
			{
				printf("FAIL: memory growth -- %ldKB vs baseline %ldKB\n", rss, baselineRss);
				return 1;
			}
		}
	}

	printf("PASS: %d create/destroy cycles, fds and RSS flat\n", iterations);
	return 0;
}

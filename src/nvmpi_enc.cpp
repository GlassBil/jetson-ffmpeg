#include "nvmpi.h"
#include "NvVideoEncoder.h"
#include "nvUtils2NvBuf.h"
#include "NVMPI_bufPool.hpp"
#include <fcntl.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <thread>
#include <unistd.h>

#ifdef WITH_CUDA_BUFFERS
#include <cuda_egl_interop.h>
#include <cuda_runtime.h>
#endif

#define MAX_BUFFERS 32
#define TEST_ERROR(condition, message, errorCode)    \
	if (condition)                               \
{                                                    \
	std::cout<< message;                         \
}

#define OUTPLANE_MEMTYPE_MMAP 0
#define OUTPLANE_MEMTYPE_DMA 1
#ifdef WITH_CUDA_BUFFERS
#define OUTPLANE_MEMTYPE OUTPLANE_MEMTYPE_DMA
#else
#define OUTPLANE_MEMTYPE OUTPLANE_MEMTYPE_MMAP
#endif

using namespace std;

struct nvmpictx
{
	uint32_t index;
	uint32_t width;
	uint32_t height;
	uint32_t profile;
	uint32_t bitrate;
	uint32_t peak_bitrate;
	uint32_t raw_pixfmt;
	uint32_t encoder_pixfmt;
	uint32_t iframe_interval;
	uint32_t idr_interval;
	uint32_t fps_n;
	uint32_t fps_d;
	uint32_t qmax;
	uint32_t qmin;
	uint32_t num_b_frames;
	uint32_t num_reference_frames;
	uint32_t vbv_buffer_size; //virtual buffer size of the encoder
	uint32_t packets_num;

	bool insert_sps_pps_at_idr;
	bool max_perf; //enable max performance mode
	bool enable_extended_colorformat;
	bool enableLossless;
	bool blocking_mode;
	bool capPlaneGotEOS;
	bool flushing;
	bool dqThreadStarted;

	enum v4l2_mpeg_video_bitrate_mode ratecontrol;
	enum v4l2_mpeg_video_h264_level level;
	enum v4l2_enc_hw_preset_type hw_preset_type;

	NvVideoEncoder *enc;
	NVMPI_bufPool<nvPacket*>* pktPool;
	int *output_plane_fd; //array to store dmabuf fd's
#ifdef WITH_CUDA_BUFFERS
	cudaEglFrame egl_frame[MAX_BUFFERS];
	cudaGraphicsResource_t egl_resource[MAX_BUFFERS];
	NvBufSurface* out_surf[MAX_BUFFERS];
#endif
};

// Frees every resource nvmpi_create_encoder allocates; safe on a partially-built context.
// Forward-declared here because create's failure paths call it (it is defined near close()).
static void destroy_encoder_context(nvmpictx* ctx);


static bool encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer, NvBuffer * shared_buffer __attribute__((unused)), void *arg)
{
	nvmpictx *ctx = (nvmpictx*)arg;

	if (v4l2_buf == NULL)
	{
		cout << "Error while dequeing buffer from output plane" << endl;
		return false;
	}

	if (buffer->planes[0].bytesused == 0)
	{
		ctx->capPlaneGotEOS = true;
		//cout << "Got 0 size buffer in capture \n"; //TODO  log it
		return false;
	}
	
	v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
	ctx->enc->getMetadata(v4l2_buf->index, enc_metadata);
	
	//nvPacket.payload --> AVPacket->data
	//nvPacket.privData --> AVPacket
	nvPacket* pkt = ctx->pktPool->dqEmptyBuf();
	if(!pkt)
	{
		//TODO wait for user to read buffer. make send_frame return AVERROR(EAGAIN) until avcodec_receive_packet() is called
		//TODO pass warning to avlog
		printf("[libnvmpi][W]: EAGAIN. User must read output. nvmpi encoder packet memory pool is empty! Packet will be dropped. There may be artifacts in the output video.\n");
	}
	else
	{
		pkt->pts = (v4l2_buf->timestamp.tv_usec % 1000000) + (v4l2_buf->timestamp.tv_sec * 1000000UL);
		//AV_PKT_FLAG_KEY 0x0001. if current packet is keyframe then enc_metadata.KeyFrame should be 0x1, so it should be OK to just assign value
		pkt->flags = enc_metadata.KeyFrame;
		pkt->payload_size = buffer->planes[0].bytesused;
		memcpy(pkt->payload, buffer->planes[0].data, pkt->payload_size);
		
		ctx->pktPool->qFilledBuf(pkt);
	}
	
	if (ctx->enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0)
	{
		//TODO error handling
		ERROR_MSG("Error while Qing buffer at capture plane");
		return false;
	}

	return true;
}

#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_DMA)
static int setup_output_dmabuf(nvmpictx *ctx, uint32_t num_buffers )
{
    int ret=0;
    NvBufSurf::NvCommonAllocateParams cParams;
    int fd;
    ret = ctx->enc->output_plane.reqbufs(V4L2_MEMORY_DMABUF,num_buffers);
    if(ret)
    {
        cerr << "reqbufs failed for output plane V4L2_MEMORY_DMABUF" << endl;
        return ret;
    }
    for (uint32_t i = 0; i < ctx->enc->output_plane.getNumBuffers(); i++)
    {
        cParams.width = ctx->width;
        cParams.height = ctx->height;
        cParams.layout = NVBUF_LAYOUT_PITCH;
        
        /*
        switch (ctx->cs)
        {
            case V4L2_COLORSPACE_REC709:
                cParams.colorFormat = ctx->enable_extended_colorformat ?
                    NVBUF_COLOR_FORMAT_YUV420_709_ER : NVBUF_COLOR_FORMAT_YUV420_709;
                break;
            case V4L2_COLORSPACE_SMPTE170M:
            default:
                cParams.colorFormat = ctx->enable_extended_colorformat ?
                    NVBUF_COLOR_FORMAT_YUV420_ER : NVBUF_COLOR_FORMAT_YUV420;
        }
        if (ctx->is_semiplanar)
        {
            cParams.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        }
        if (ctx->encoder_pixfmt == V4L2_PIX_FMT_H264)
        {
            if (ctx->enableLossless)
            {
                if (ctx->is_semiplanar)
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_NV24;
                else
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_YUV444;
            }
        }
        
        
        else if (ctx->encoder_pixfmt == V4L2_PIX_FMT_H265)
        {
            if (ctx->chroma_format_idc == 3)
            {
                if (ctx->is_semiplanar)
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_NV24;
                else
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_YUV444;

                if (ctx->bit_depth == 10)
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_NV24_10LE;
            }
            if (ctx->profile == V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10 && (ctx->bit_depth == 10))
            {
                cParams.colorFormat = NVBUF_COLOR_FORMAT_NV12_10LE;
            }
        }
        */
        
        cParams.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
        cParams.memtag = NvBufSurfaceTag_VIDEO_ENC;
        cParams.memType = NVBUF_MEM_SURFACE_ARRAY;
        // Create output plane fd for DMABUF io-mode
        ret = NvBufSurf::NvAllocate(&cParams, 1, &fd);
        if(ret < 0)
        {
            cerr << "Failed to create NvBuffer" << endl;
            return ret;
        }
        ctx->output_plane_fd[i]=fd;
#ifdef WITH_CUDA_BUFFERS
        {
            NvBufSurface *surf = nullptr;
            if (NvBufSurfaceFromFd(fd, (void**)&surf) != 0)
            {
                cerr << "NvBufSurfaceFromFd failed for output buffer " << i << endl;
                return -1;
            }
            ctx->out_surf[i] = surf;

            cudaSetDevice(0);
            if (surf->surfaceList[0].mappedAddr.eglImage == NULL)
            {
                if (NvBufSurfaceMapEglImage(surf, 0) != 0)
                {
                    cerr << "Unable to map EGL Image for output buffer " << i << endl;
                    return -1;
                }
            }

            EGLImageKHR egl_image = surf->surfaceList[0].mappedAddr.eglImage;
            if (egl_image == nullptr)
            {
                cerr << "Null EGLImage for output buffer " << i << endl;
                return -1;
            }

            cudaFree(0); // ensure a CUDA context exists on this thread
            cudaError_t cuda_status = cudaGraphicsEGLRegisterImage(&ctx->egl_resource[i], egl_image, cudaGraphicsRegisterFlagsNone);
            if (cuda_status != cudaSuccess)
            {
                cerr << "cudaGraphicsEGLRegisterImage failed for output buffer " << i << ": " << cudaGetErrorName(cuda_status) << endl;
                return -1;
            }

            cuda_status = cudaGraphicsResourceGetMappedEglFrame(&ctx->egl_frame[i], ctx->egl_resource[i], 0, 0);
            if (cuda_status != cudaSuccess)
            {
                cerr << "cudaGraphicsResourceGetMappedEglFrame failed for output buffer " << i << ": " << cudaGetErrorName(cuda_status) << endl;
                return -1;
            }
        }
#endif
    }
    return ret;
}
#endif

nvmpictx* nvmpi_create_encoder(nvEncParam* param)
{
	int ret;
	log_level = LOG_LEVEL_INFO;
	//log_level = LOG_LEVEL_DEBUG;
	nvmpictx *ctx=new nvmpictx;
	ctx->index=0;
	ctx->width=param->width;
	ctx->height=param->height;
	ctx->enableLossless=false;
	ctx->bitrate=param->bitrate;
	ctx->ratecontrol = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;	
	ctx->idr_interval = param->idr_interval;
	ctx->fps_n = param->fps_n;
	ctx->fps_d = param->fps_d;
	ctx->iframe_interval = param->iframe_interval;
	ctx->pktPool = new NVMPI_bufPool<nvPacket*>();
	ctx->enable_extended_colorformat=false;
	ctx->packets_num=param->capture_num;
#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_DMA)
	ctx->output_plane_fd = new int[ctx->packets_num];
	for(uint32_t i = 0; i < ctx->packets_num; i++)
	{
		ctx->output_plane_fd[i] = -1;
	}
#endif
#ifdef WITH_CUDA_BUFFERS
	for(int i = 0; i < MAX_BUFFERS; i++)
	{
		ctx->egl_resource[i] = nullptr;
		ctx->out_surf[i] = nullptr;
	}
#endif
	ctx->qmax=param->qmax;
	ctx->qmin=param->qmin;
	ctx->num_b_frames=param->max_b_frames;
	ctx->num_reference_frames=param->refs;
	ctx->insert_sps_pps_at_idr=(param->insert_spspps_idr==1)?true:false;
	ctx->capPlaneGotEOS = false;
	ctx->flushing = false;
	ctx->dqThreadStarted = false;
	ctx->blocking_mode = true; //TODO non-blocking mode support
	ctx->max_perf = true; //TODO invistigate why encoder is slow without max_perf even with MAXN power mode
	ctx->vbv_buffer_size = param->vbv_buffer_size;
	
	switch(param->profile)
	{
		case 77://FF_PROFILE_H264_MAIN
			ctx->profile=V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
			break;
		case 66://FF_PROFILE_H264_BASELINE
			ctx->profile=V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
			break;
		case 100://FF_PROFILE_H264_HIGH
			ctx->profile=V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
			break;

		default:
			ctx->profile=V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
			break;
	}

	switch(param->level)
	{
		case 10:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_1_0;
			break;
		case 11:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_1_1;
			break;
		case 12:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_1_2;
			break;
		case 13:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_1_3;
			break;
		case 20:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_2_0;
			break;
		case 21:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_2_1;
			break;
		case 22:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_2_2;
			break;
		case 30:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_3_0;
			break;
		case 31:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_3_1;
			break;
		case 32:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_3_2;
			break;
		case 40:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_4_0;
			break;
		case 41:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_4_1;
			break;
		case 42:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_4_2;
			break;
		case 50:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_5_0;
			break;
		case 51:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
			break;
		default:
			ctx->level=V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
			break;	
	}

	switch(param->hw_preset_type)
	{
		case 1:
			ctx->hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
			break;
		case 2:
			ctx->hw_preset_type = V4L2_ENC_HW_PRESET_FAST;
			break;
		case 3:
			ctx->hw_preset_type = V4L2_ENC_HW_PRESET_MEDIUM;
			break;
		case 4:
			ctx->hw_preset_type = V4L2_ENC_HW_PRESET_SLOW;
			break;
		default:
			ctx->hw_preset_type = V4L2_ENC_HW_PRESET_MEDIUM;
			break;
	}

	if(param->enableLossless)
		ctx->enableLossless=true;

	if(param->mode_vbr)
		ctx->ratecontrol=V4L2_MPEG_VIDEO_BITRATE_MODE_VBR;

	if(param->codingType==NV_VIDEO_CodingH264)
	{
		ctx->encoder_pixfmt=V4L2_PIX_FMT_H264;
	}else if(param->codingType==NV_VIDEO_CodingHEVC)
	{
		ctx->encoder_pixfmt=V4L2_PIX_FMT_H265;
	}
	if(ctx->blocking_mode)
	{
		ctx->enc=NvVideoEncoder::createVideoEncoder("enc0");
	}
	else
	{
		ctx->enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK);
	}
	if(!ctx->enc)
	{
		cerr << "Could not create encoder" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	ret = ctx->enc->setCapturePlaneFormat(ctx->encoder_pixfmt, ctx->width,ctx->height, NVMPI_ENC_CHUNK_SIZE);

	if(ret < 0)
	{
		cerr << "Could not set capture plane format" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	switch (ctx->profile)
	{
		case V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10:
			ctx->raw_pixfmt = V4L2_PIX_FMT_P010M;
			break;
		case V4L2_MPEG_VIDEO_H265_PROFILE_MAIN:
		default:
			ctx->raw_pixfmt = V4L2_PIX_FMT_YUV420M;
	}

	if (ctx->enableLossless && param->codingType == NV_VIDEO_CodingH264)
	{
		ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE;
		ret = ctx->enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV444M, ctx->width,ctx->height);
	}
	else
	{
		ret = ctx->enc->setOutputPlaneFormat(ctx->raw_pixfmt, ctx->width,ctx->height);
	}

	if(ret < 0)
	{
		cerr << "Could not set output plane format" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	ret = ctx->enc->setBitrate(ctx->bitrate);
	TEST_ERROR(ret < 0, "Could not set encoder bitrate", ret);

	if(ctx->vbv_buffer_size)
	{
		/* Set virtual buffer size value for encoder */
		ret = ctx->enc->setVirtualBufferSize(ctx->vbv_buffer_size);
		TEST_ERROR(ret < 0, "Could not set virtual buffer size", ret);
	}

	ret=ctx->enc->setHWPresetType(ctx->hw_preset_type);
	TEST_ERROR(ret < 0, "Could not set encoder HW Preset Type", ret);

	if(ctx->num_reference_frames)
	{
		ret = ctx->enc->setNumReferenceFrames(ctx->num_reference_frames);
		TEST_ERROR(ret < 0, "Could not set num reference frames", ret);
	}

	if(ctx->num_b_frames != (uint32_t) -1 && param->codingType == NV_VIDEO_CodingH264)
	{
		ret = ctx->enc->setNumBFrames(ctx->num_b_frames);
		TEST_ERROR(ret < 0, "Could not set number of B Frames", ret);
	}


	if(param->codingType == NV_VIDEO_CodingH264 || param->codingType == NV_VIDEO_CodingHEVC)
	{
		ret = ctx->enc->setProfile(ctx->profile);
		TEST_ERROR(ret < 0, "Could not set encoder profile", ret);
	}

	if(param->codingType== NV_VIDEO_CodingH264)
	{
		ret = ctx->enc->setLevel(ctx->level);
		TEST_ERROR(ret < 0, "Could not set encoder level", ret);
	}


	if (ctx->enableLossless)
	{
		ret = ctx->enc->setConstantQp(0);
		TEST_ERROR(ret < 0, "Could not set encoder constant qp=0", ret);

	}
	else
	{
		ret = ctx->enc->setRateControlMode(ctx->ratecontrol);
		TEST_ERROR(ret < 0, "Could not set encoder rate control mode", ret);

		if (ctx->ratecontrol == V4L2_MPEG_VIDEO_BITRATE_MODE_VBR)
		{
			uint32_t peak_bitrate;
			//TODO log warning?
			if (ctx->peak_bitrate < ctx->bitrate)
				peak_bitrate = 1.2f * ctx->bitrate;
			else
				peak_bitrate = ctx->peak_bitrate;
			ret = ctx->enc->setPeakBitrate(peak_bitrate);
			TEST_ERROR(ret < 0, "Could not set encoder peak bitrate", ret);
		}
	}

	ret = ctx->enc->setIDRInterval(ctx->idr_interval);
	TEST_ERROR(ret < 0, "Could not set encoder IDR interval", ret);

	if(ctx->qmax>0 ||ctx->qmin >0){
		ctx->enc->setQpRange(ctx->qmin, ctx->qmax, ctx->qmin,ctx->qmax, ctx->qmin, ctx->qmax);	
	}
	ret = ctx->enc->setIFrameInterval(ctx->iframe_interval);
	TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", ret);
	
    if(ctx->max_perf)
    {
        /* Enable maximum performance mode by disabling internal DFS logic.
           NOTE: This enables encoder to run at max clocks */
		ret = ctx->enc->setMaxPerfMode(ctx->max_perf);
		TEST_ERROR(ret < 0, "Error while setting encoder to max perf", ret);
	}
	
	if(ctx->insert_sps_pps_at_idr){
		ret = ctx->enc->setInsertSpsPpsAtIdrEnabled(true);
		TEST_ERROR(ret < 0, "Could not set insertSPSPPSAtIDR", ret);
	}

	ret = ctx->enc->setFrameRate(ctx->fps_n, ctx->fps_d);
	TEST_ERROR(ret < 0, "Could not set framerate", ret);
	
	//ret = ctx->enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, ctx->packets_num, false, true);
#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_MMAP)
	ret = ctx->enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, ctx->packets_num, true, false);
#else
	ret = setup_output_dmabuf(ctx,ctx->packets_num); //V4L2_MEMORY_DMABUF
#endif
	if(ret < 0)
	{
		cerr << "Could not setup output plane" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	ret = ctx->enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, ctx->packets_num, true, false);
	if(ret < 0)
	{
		cerr << "Could not setup capture plane" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	ret = ctx->enc->subscribeEvent(V4L2_EVENT_EOS,0,0);
	if(ret < 0)
	{
		cerr << "Could not subscribe EOS event" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	ret = ctx->enc->output_plane.setStreamStatus(true);
	if(ret < 0)
	{
		cerr << "Error in output plane streamon" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	ret = ctx->enc->capture_plane.setStreamStatus(true);
	if(ret < 0)
	{
		cerr << "Error in capture plane streamon" << endl;
		destroy_encoder_context(ctx);
		return nullptr;
	}

	if(ctx->blocking_mode)
	{
		ctx->enc->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback);
		ctx->enc->capture_plane.startDQThread(ctx);
		ctx->dqThreadStarted = true;
	}
    else
    {
		/*
        sem_init(&ctx->pollthread_sema, 0, 0);
        sem_init(&ctx->encoderthread_sema, 0, 0);
        // Set encoder poll thread for non-blocking io mode
        pthread_create(&ctx->enc_pollthread, NULL, encoder_pollthread_fcn, ctx);
        pthread_setname_np(ctx->enc_pollthread, "EncPollThread");
        cout << "Created the PollThread and Encoder Thread \n";
        */
    }

	// Enqueue all the empty capture plane buffers
	for (uint32_t i = 0; i < ctx->enc->capture_plane.getNumBuffers(); i++){
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];
		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

		v4l2_buf.index = i;
		v4l2_buf.m.planes = planes;

		ret = ctx->enc->capture_plane.qBuffer(v4l2_buf, NULL);
		if(ret < 0)
		{
			cerr << "Error while queueing buffer at capture plane" << endl;
			destroy_encoder_context(ctx);
			return nullptr;
		}

	}

	return ctx;
}

int copyFrameToNvBuf(nvFrame* frame, NvBuffer& buffer)
{
	uint32_t i, j;
	char *dataDst;
	char *dataSrc;
	for (i = 0; i < buffer.n_planes; i++)
	{
		NvBuffer::NvBufferPlane &plane = buffer.planes[i];
		unsigned int &frameLineSize = frame->linesize[i];
		size_t copySz = plane.fmt.bytesperpixel * plane.fmt.width;
		dataDst = (char *) plane.data;
		dataSrc = (char *) frame->payload[i];
		plane.bytesused = 0;
		for (j = 0; j < plane.fmt.height; j++)
		{
			memcpy(dataDst, dataSrc, copySz);
			dataDst += plane.fmt.stride;
			dataSrc += frameLineSize;
		}
		plane.bytesused = plane.fmt.stride * plane.fmt.height;
	}
	return 0;
}

#ifdef WITH_CUDA_BUFFERS
int copyFrameToNvBufCuda(nvmpictx* ctx, nvFrame* frame, uint32_t bufIndex)
{
	cudaEglFrame &eglFrame = ctx->egl_frame[bufIndex];

	// Detect whether the caller's buffers live in device or host memory.
	// Device/managed -> D2D; unregistered host (e.g. the extradata dummy
	// frame allocated with av_image_alloc) -> H2D.
	enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
	cudaPointerAttributes attr;
	cudaError_t st = cudaPointerGetAttributes(&attr, frame->payload[0]);
	if (st == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged))
	{
		kind = cudaMemcpyDeviceToDevice;
	}
	else
	{
		cudaGetLastError(); // clear the error left by an unregistered host pointer
	}

	// Drive the copy from the surface's own plane geometry rather than assuming a tightly-packed
	// 3-plane YUV420 layout. The destination pitch must be the surface pitch (what the encoder
	// reads), which is alignment-padded and not the tightly-packed width; the plane count, per-plane
	// row-byte count and height also come from the surface. This mirrors how the decoder reads these
	// buffers (see nvmpi_dec.cpp) and avoids the sheared/striped output and unwritten trailing rows a
	// pitch/height mismatch produces.
	NvBufSurfacePlaneParams &parm = ctx->out_surf[bufIndex]->surfaceList[0].planeParams;
	for (uint32_t plane = 0; plane < parm.num_planes; plane++)
	{
		void *dst = eglFrame.frame.pPitch[plane].ptr;
		size_t dstPitch = parm.pitch[plane];
		void *src = frame->payload[plane];
		size_t srcPitch = frame->linesize[plane];
		size_t widthBytes = parm.width[plane] * parm.bytesPerPix[plane];
		size_t height = parm.height[plane];

		cudaError_t cuda_status = cudaMemcpy2D(dst, dstPitch, src, srcPitch, widthBytes, height, kind);
		if (cuda_status != cudaSuccess)
		{
			cerr << "CUDA memcpy failed for plane " << plane << ": " << cudaGetErrorString(cuda_status) << endl;
			return -1;
		}
	}

	NvBufSurfaceSyncForDevice(ctx->out_surf[bufIndex], 0, -1);
	return 0;
}
#endif

int nvmpi_encoder_put_frame(nvmpictx* ctx,nvFrame* frame)
{
	if(ctx->flushing) return -2;
	
	int ret;
	struct v4l2_buffer v4l2_buf;
	struct v4l2_plane planes[MAX_PLANES];
	NvBuffer *nvBuffer;

	memset(&v4l2_buf, 0, sizeof(v4l2_buf));
	memset(planes, 0, sizeof(planes));

	v4l2_buf.m.planes = planes;

	if(ctx->enc->isInError())
		return -1;

	if(ctx->index < ctx->enc->output_plane.getNumBuffers())
	{
		nvBuffer=ctx->enc->output_plane.getNthBuffer(ctx->index);
		v4l2_buf.index = ctx->index;
		ctx->index++;
		
#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_DMA)
		v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
		v4l2_buf.memory = V4L2_MEMORY_DMABUF;
		// Map output plane buffer for memory type DMABUF.
		ret = ctx->enc->output_plane.mapOutputBuffers(v4l2_buf, ctx->output_plane_fd[v4l2_buf.index]);
		if (ret < 0)
		{
			cerr << "Error while mapping buffer at output plane" << endl;
		}
#endif
	}
	else
	{
		/*TODO move it to another thread or make this call non-blocking.
		 * DQBuffer could take considerable time. e.g. when encoder performance is 20fps and all output_plane is busy
		 * it could take up to 1000/20=50ms... latency
		 */
		ret = ctx->enc->output_plane.dqBuffer(v4l2_buf, &nvBuffer, NULL, -1);
		if (ret < 0)
		{
			cout << "Error DQing buffer at output plane" << std::endl;
			return false;
		}
	}
	
	if(frame)
	{
#ifdef WITH_CUDA_BUFFERS
		copyFrameToNvBufCuda(ctx, frame, v4l2_buf.index);
		// copyFrameToNvBufCuda writes the EGL-mapped surface but not the NvBuffer plane sizes.
		// The DMABUF qBuffer below takes bytesused from nvBuffer->planes[].bytesused, and a
		// zero there is the encoder's EOS marker, so it must carry the real frame size.
		for (uint32_t j = 0; j < nvBuffer->n_planes; j++)
		{
			nvBuffer->planes[j].bytesused = nvBuffer->planes[j].fmt.stride * nvBuffer->planes[j].fmt.height;
		}
#else
		copyFrameToNvBuf(frame, *nvBuffer);
#endif
		v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
		v4l2_buf.timestamp.tv_usec = frame->timestamp % 1000000;
		v4l2_buf.timestamp.tv_sec = frame->timestamp / 1000000;
	}
	else
	{
		//send EOS and flush
		ctx->flushing = true;
		v4l2_buf.m.planes[0].m.userptr = 0;
		v4l2_buf.m.planes[0].bytesused = v4l2_buf.m.planes[1].bytesused = v4l2_buf.m.planes[2].bytesused = 0;
#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_DMA)
		// The DMABUF bytesused copy below sources from nvBuffer->planes[]; a stale non-zero size left
		// by the previous frame would mask this EOS marker, so clear it on the buffer object too.
		for (uint32_t j = 0; j < nvBuffer->n_planes; j++)
		{
			nvBuffer->planes[j].bytesused = 0;
		}
#endif
	}

	//needed for V4L2_MEMORY_MMAP and V4L2_MEMORY_DMABUF
	for (uint32_t j = 0 ; j < nvBuffer->n_planes; j++)
	{
#ifdef WITH_NVUTILS
		NvBufSurface *nvbuf_surf = 0;
		ret = NvBufSurfaceFromFd (nvBuffer->planes[j].fd, (void**)(&nvbuf_surf));
		if (ret < 0)
		{
			cerr << "Error while NvBufSurfaceFromFd" << endl;
		}
		ret = NvBufSurfaceSyncForDevice (nvbuf_surf, 0, j);
		if (ret < 0)
		{
			cerr << "Error while NvBufSurfaceSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
		}
#else
		ret = NvBufferMemSyncForDevice (nvBuffer->planes[j].fd, j, (void **)&nvBuffer->planes[j].data);
		if (ret < 0)
		{
			cerr << "Error while NvBufferMemSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
		}
#endif
	}
	
	//for V4L2_MEMORY_DMABUF only
#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_DMA)
	for (uint32_t j = 0; j < nvBuffer->n_planes; j++)
	{
		v4l2_buf.m.planes[j].bytesused = nvBuffer->planes[j].bytesused;
	}
#endif

	ret = ctx->enc->output_plane.qBuffer(v4l2_buf, NULL);
	TEST_ERROR(ret < 0, "Error while queueing buffer at output plane", ret);

	return 0;
}

int nvmpi_encoder_dqEmptyPacket(nvmpictx* ctx,nvPacket** packet)
{
	nvPacket* pkt = ctx->pktPool->dqEmptyBuf();
	if(!pkt) return -1;
	*packet = pkt;
	return 0;
}

void nvmpi_encoder_qEmptyPacket(nvmpictx* ctx,nvPacket* packet)
{
	ctx->pktPool->qEmptyBuf(packet);
	return;
}

int nvmpi_encoder_get_packet(nvmpictx* ctx,nvPacket** packet)
{
	nvPacket* pkt = ctx->pktPool->dqFilledBuf();
	
	if(!pkt)
	{
		if(!ctx->flushing) return -1;
		bool wait = true;
		while(wait)
		{
			pkt = ctx->pktPool->dqFilledBuf();
			if(pkt || ctx->capPlaneGotEOS) wait = false;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		if(!pkt) return -2; //if got eos
	}
	
	*packet = pkt;
	return 0;
}

// Free every resource nvmpi_create_encoder allocates. Safe on a fully- or partially-built
// context: each resource is null/-1 guarded, the cleanup is driven off the packets_num-sized
// arrays (not output_plane.getNumBuffers(), which may be 0 on a partial build), it never
// early-returns, and it deletes ctx last. Called by nvmpi_encoder_close and by every
// nvmpi_create_encoder failure path.
static void destroy_encoder_context(nvmpictx* ctx)
{
	if(!ctx)
	{
		return;
	}

	// Stop the capture DQ thread before freeing anything it might touch. abort() unblocks a
	// thread parked in a blocking dqBuffer so the join cannot time out and race delete enc.
	if(ctx->enc && ctx->dqThreadStarted && ctx->blocking_mode)
	{
		ctx->enc->abort();
		ctx->enc->capture_plane.stopDQThread();
		ctx->enc->capture_plane.waitForDQThread(-1);
		ctx->dqThreadStarted = false;
	}

#if (OUTPLANE_MEMTYPE == OUTPLANE_MEMTYPE_DMA)
	if(ctx->output_plane_fd)
	{
		for(uint32_t i = 0; i < ctx->packets_num; i++)
		{
#ifdef WITH_CUDA_BUFFERS
			if(ctx->egl_resource[i])
			{
				cudaGraphicsUnregisterResource(ctx->egl_resource[i]);
				ctx->egl_resource[i] = nullptr;
			}

			if(ctx->out_surf[i] && ctx->out_surf[i]->surfaceList[0].mappedAddr.eglImage)
			{
				NvBufSurfaceUnMapEglImage(ctx->out_surf[i], 0);
			}

			ctx->out_surf[i] = nullptr;
#endif
			if(ctx->output_plane_fd[i] != -1)
			{
				if(ctx->enc)
				{
					ctx->enc->output_plane.unmapOutputBuffers(i, ctx->output_plane_fd[i]);
				}

				NvBufSurf::NvDestroy(ctx->output_plane_fd[i]);
				ctx->output_plane_fd[i] = -1;
			}
		}

		delete[] ctx->output_plane_fd;
		ctx->output_plane_fd = nullptr;
	}
#endif

	delete ctx->enc;
	delete ctx->pktPool;
	delete ctx;
}

int nvmpi_encoder_close(nvmpictx* ctx)
{
	destroy_encoder_context(ctx);
	return 0;
}


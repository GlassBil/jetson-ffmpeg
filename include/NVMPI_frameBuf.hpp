#pragma once
#include "nvUtils2NvBuf.h"
#ifdef WITH_CUDA_BUFFERS
#include <cuda_egl_interop.h>
#include <cuda_runtime.h>
#endif

struct NVMPI_frameBuf
{
#ifdef WITH_NVUTILS
	NvBufSurface *dst_dma_surface = NULL;
#endif
	int dst_dma_fd = -1;
	unsigned long long timestamp = 0;

#ifdef WITH_CUDA_BUFFERS
	void *cuda_ptr[3] = {nullptr, nullptr, nullptr};
	size_t cuda_pitch[3] = {0, 0, 0};
    cudaEglFrame egl_frame;
    cudaGraphicsResource_t egl_resource{nullptr};
#endif
	
	//allocate DMA buffer
	bool alloc(NvBufferCreateParams& input_params);
	//destroy DMA buffer
	bool destroy();
};

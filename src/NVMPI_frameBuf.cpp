#include "NVMPI_frameBuf.hpp"
#include <iostream> //LOG. TODO: add some LOG() define

bool NVMPI_frameBuf::alloc(NvBufferCreateParams& input_params)
{
	int ret = 0;
#ifdef WITH_NVUTILS
	ret = NvBufSurf::NvAllocate(&input_params, 1, &dst_dma_fd);
	if(ret<0)
	{
		std::cout << "Failed to allocate buffer" << std::endl;
		return false;
	}
	
	ret = NvBufSurfaceFromFd(dst_dma_fd, (void**)(&dst_dma_surface));
	if(ret<0)
	{
		std::cout << "Failed to get surface for buffer" << std::endl;
		NvBufferDestroy(dst_dma_fd);
		dst_dma_fd = -1;
		return false;
	}

#ifdef WITH_CUDA_BUFFERS
	cudaSetDevice(0);
	if (dst_dma_surface->surfaceList[0].mappedAddr.eglImage == NULL)
	{
		if (NvBufSurfaceMapEglImage(dst_dma_surface, 0) != 0)
		{
			std::cout << "Unable to map EGL Image" << std::endl;
			destroy();
			return false;
		}
	}

	EGLImageKHR egl_image = dst_dma_surface->surfaceList[0].mappedAddr.eglImage;
	if (egl_image == nullptr)
	{
		std::cout << "Error while mapping dmabuf fd (" << dst_dma_fd << ") to EGLImage" << std::endl;
		destroy();
		return false;
	}

	cudaFree(0);
	cudaError_t cuda_status = cudaGraphicsEGLRegisterImage(&egl_resource, egl_image, cudaGraphicsRegisterFlagsReadOnly);
	if (cuda_status != cudaSuccess)
	{
		std::cout << "cudaGraphicsEGLRegisterImage failed: " << cudaGetErrorName(cuda_status) << std::endl;
		destroy();
		return false;
	}

	cuda_status = cudaGraphicsResourceGetMappedEglFrame(&egl_frame, egl_resource, 0, 0);
	if (cuda_status != cudaSuccess)
	{
		std::cout << "cudaGraphicsResourceGetMappedEglFrame failed: " << cudaGetErrorName(cuda_status) << std::endl;
		destroy();
		return false;
	}

	switch(input_params.colorFormat)
	{
		case NVBUF_COLOR_FORMAT_NV12:
		case NVBUF_COLOR_FORMAT_NV12_ER:
		case NVBUF_COLOR_FORMAT_NV12_709:
		case NVBUF_COLOR_FORMAT_NV12_709_ER:
		case NVBUF_COLOR_FORMAT_NV12_2020:
		{
			for (int i = 0; i < 2; i++)
			{
				cuda_status = cudaMallocPitch(&cuda_ptr[i], &cuda_pitch[i], input_params.width,
					(i == 0) ? input_params.height : input_params.height / 2);
				if(cuda_status != cudaSuccess)
				{
					std::cout << "Failed to allocate CUDA memory for plane " << i << ": " << cudaGetErrorString(cuda_status) << std::endl;
					destroy();
					return false;
				}
			}
			break;
		}
		case NVBUF_COLOR_FORMAT_YUV420:
		{
			for (int i = 0; i < 3; i++)
			{
				cuda_status = cudaMallocPitch(&cuda_ptr[i], &cuda_pitch[i], input_params.width / (i == 0 ? 1 : 2),
					input_params.height / (i == 0 ? 1 : 2));
				if(cuda_status != cudaSuccess)
				{
					std::cout << "Failed to allocate CUDA memory for plane " << i << ": " << cudaGetErrorString(cuda_status) << std::endl;
					destroy();
					return false;
				}
			}
			break;
		}
		default:
		{
			std::cout << "Unsupported color format for CUDA allocation" << std::endl;
			destroy();
			return false;
		}
	}
#endif

#else
	ret = NvBufferCreateEx(&dst_dma_fd, &input_params);
	if(ret<0)
	{
		std::cout << "Failed to allocate buffer" << std::endl;
		return false;
	}
#endif
	
	return true;
}

bool NVMPI_frameBuf::destroy()
{
	bool success = true;

#ifdef WITH_CUDA_BUFFERS
	if(egl_resource)
	{
		cudaGraphicsUnregisterResource(egl_resource);
		egl_resource = nullptr;
	}
	if(dst_dma_surface && dst_dma_surface->surfaceList[0].mappedAddr.eglImage)
	{
		NvBufSurfaceUnMapEglImage(dst_dma_surface, 0);
		dst_dma_surface->surfaceList[0].mappedAddr.eglImage = nullptr;
	}

	for(int i = 0; i < 3; i++)
	{
		if(cuda_ptr[i] != nullptr)
		{
			cudaError_t cuda_status = cudaFree(cuda_ptr[i]);
			if(cuda_status != cudaSuccess)
			{
				std::cout << "Failed to free CUDA memory for plane " << i << ": " << cudaGetErrorString(cuda_status) << std::endl;
				success = false;
			}
			cuda_ptr[i] = nullptr;
			cuda_pitch[i] = 0;
		}
	}
#endif

	int ret = 0;
	if(dst_dma_fd >= 0)
	{
		ret = NvBufferDestroy(dst_dma_fd);
		if(ret<0)
		{
			std::cout << "Failed to Destroy NvBuffer" << std::endl;
			success = false;
		}
		dst_dma_fd = -1;
#ifdef WITH_NVUTILS
		dst_dma_surface = NULL;
#endif
	}

	return success;
}

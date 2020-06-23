#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "vector_td_utilities.h"
#include "real_utilities.h"
#include "real_utilities_device.h"
#include "complext.h"
#include "check_CUDA.h"
#include "cudaDeviceManager.h"
#include "setup_grid.h"
#include "GPUTimer.h"
#include <iostream>
#include <cmath>
#include "CUBLASContextProvider.h"
#include <cublas_v2.h>
#include "test_gpu_benoit.h"

namespace Gadgetron{

    __global__ void cuZerofilling(complext<float> *data_in, complext<float> *data_out, int *in_dimensions, int *out_dimensions, int offsetX, int offsetY)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;

            if (i < in_dimensions[0] * in_dimensions[1])
            {
                data_out[(blockIdx.x + offsetX) * out_dimensions[0] + threadIdx.x] = data_in[i];
            }
    }

    void execute_zerofilling_gpu(complext<float> *data_in, complext<float> *data_out, int *in_dimensions, int *out_dimensions, int offsetX, int offsetY, size_t sizeX, size_t sizeY)
    {
        GDEBUG("-------------------ENTERING EXECUTE_ZEROFILLING_GPU--------------------------\n");
        int max_thread_x_size = 1024;
        int max_block_x_size = (sizeX * sizeY) / max_thread_x_size;

        cuZerofilling<<<max_block_x_size, max_thread_x_size>>>(data_in, data_out, in_dimensions, out_dimensions, offsetX, offsetY);
        GDEBUG("-------------------LEAVING EXECUTE_ZEROFILLING_GPU--------------------------\n");
    }
}
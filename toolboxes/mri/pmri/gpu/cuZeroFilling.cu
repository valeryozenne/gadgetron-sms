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

    // __global__ void image_copy_center(uchar *src, uchar *dest, int rowSize, int colSize, int newRowSize, int newColSize)
    // {
    // int indexX = threadIdx.x + blockIdx.x * blockDim.x;//X : colSize
    // int indexY = threadIdx.y + blockIdx.y * blockDim.y;//Y : rowSize

    // int newIndexX = indexX + offsetX;
    // int newIndexY = indexY + offsetY;


    // if (indexX < colSize && indexY < rowSize)
    //     dest[newIndexX * newColSize + newIndexY] = src[indexX * colSize + indexY];
    // }

    __global__ void cuZerofilling(complext<float> *data_in, complext<float> *data_out, int rowSize, int colSize, int outRowSize, int outColSize, int offsetX, int offsetY)
    {
        int indexX = threadIdx.x + blockIdx.x * blockDim.x;//X : colSize
        int indexY = threadIdx.y + blockIdx.y * blockDim.y;//Y : rowSize

        int newIndexX = indexX + offsetX;
        int newIndexY = indexY + offsetY;

        
        if (indexX < colSize && indexY < rowSize)
        {
            data_out[newIndexX * outColSize + newIndexX] = data_in[indexX * colSize + indexY];
        }
    }

    void execute_zerofilling_gpu(complext<float> *data_in, complext<float> *data_out, int *in_dimensions, int *out_dimensions, int offsetX, int offsetY, size_t sizeX, size_t sizeY)
    {
        // int max_thread_x_size = 1024;
        // int max_block_x_size = (sizeX * sizeY) / max_thread_x_size;
        // GDEBUG("MAX THREAD SIZE: %d\n", max_thread_x_size);
        // GDEBUG("MAX BLOCK SIZE: %d\n", max_block_x_size);

        int rowSize = in_dimensions[0];
        int colSize = in_dimensions[1];

        int outRowSize = out_dimensions[0];
        int outColSize = out_dimensions[1];

        dim3 threadsPerBlock(1, 1);
        
        dim3 numBlocks(rowSize/threadsPerBlock.x, colSize/threadsPerBlock.y); 
        
        std::cout << "Threads per block : ["  << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << "]" << std::endl;
        std::cout << "Number of blocks : [" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << "]" << std::endl;
        
        //image_copy_center<<<numBlocks, threadsPerBlock>>>(dev_data, dev_out_data, rowSize, colSize, outRowSize, outColSize);

        cuZerofilling<<<numBlocks, threadsPerBlock>>>(data_in, data_out, in_dimensions[0], in_dimensions[1], out_dimensions[0], out_dimensions[1], offsetX, offsetY);
    }
}
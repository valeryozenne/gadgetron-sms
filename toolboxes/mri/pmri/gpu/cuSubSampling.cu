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
#include "cuZeroFilling.h"

namespace Gadgetron{

    template <class T>
    __global__ void subsampling_2D (const T *data_in, T *data_out_level_1, T *data_out_level_2, T *data_out_level_3, T *data_out_level_4, int RO, int E1, int SLC)
    {
        int ro = threadIdx.x + blockIdx.x * blockDim.x; //X : rowIdx
        int e1 = threadIdx.y + blockIdx.y * blockDim.y; //Y : colIdx

        int ROsubsampling1 = RO / 2;//size of first subsampling array
        int E1subsampling1 = E1 / 2;
        int ROsubsampling2 = RO / 4;//size of second subsampling array
        int E1subsampling2 = E1 / 4;
        int ROsubsampling3 = RO / 8;//size of third subsampling array
        int E1subsampling3 = E1 / 8;

        if (ro < RO && e1 < E1)
        {
            data_out_level_1[ro * E1 + e1] = data_in[ro * E1 + e1];
            if (ro < ROsubsampling1 && e1 < E1subsampling1)
            {
                data_out_level_2[ro * E1subsampling1 + e1] = data_in[ro * E1 + e1];
                if (ro < ROsubsampling2 && e1 < E1subsampling2)
                {
                    data_out_level_3[ro * E1subsampling2 + e1] = data_in[ro * E1 + e1];
                    if (ro < ROsubsampling3 && e1 < E1subsampling3)
                    {
                        data_out_level_4[ro * E1subsampling3 + e1] = data_in[ro * E1 + e1];
                    }
                }
            }
        }
        
    }

    template <class T>
    __global__ void subsampling_3D (const T *data_in, T *data_out_level_1, T *data_out_level_2, T *data_out_level_3, T *data_out_level_4, int RO, int E1, int SLC)
    {
        int ro = threadIdx.x + blockIdx.x * blockDim.x; //X : rowIdx
        int e1 = threadIdx.y + blockIdx.y * blockDim.y; //Y : colIdx
        int slc = blockIdx.z;
        
        int ROsubsampling1 = RO / 2;//size of first subsampling array
        int E1subsampling1 = E1 / 2;
        int ROsubsampling2 = RO / 4;//size of second subsampling array
        int E1subsampling2 = E1 / 4;
        int ROsubsampling3 = RO / 8;//size of third subsampling array
        int E1subsampling3 = E1 / 8;

        if (ro < RO && e1 < E1 && slc < SLC)
        {
            data_out_level_1[slc * RO * E1 + ro * E1 + e1] = data_in[slc * RO * E1 + ro * E1 + e1];
            if (ro < ROsubsampling1 && e1 < E1subsampling1)
            {
                data_out_level_2[slc * ROsubsampling1 * E1subsampling1 + ro * E1subsampling1 + e1] = data_in[slc * RO * E1 + ro * E1 + e1];
                if (ro < ROsubsampling2 && e1 < E1subsampling2)
                {
                    data_out_level_3[slc * ROsubsampling2 * E1subsampling2 + ro * E1subsampling2 + e1] = data_in[slc * RO * E1 + ro * E1 + e1];
                    if (ro < ROsubsampling3 && e1 < E1subsampling3)
                    {
                        data_out_level_4[slc * ROsubsampling3 * E1subsampling3 + ro * E1subsampling3 + e1] = data_in[slc * RO * E1 + ro * E1 + e1];
                    }
                }
            }
        }
		
    }

    template <class T> void 
    execute_subsampling_2D(T *data_in, T *data_out_level_0, T *data_out_level_1, T *data_out_level_2, T *data_out_level_3, int RO, int E1, int SLC)
    {
        dim3 threadsPerBlock(32, 32, 1);
            
        unsigned int nbBlocksRO = (unsigned int )(std::ceil((float)((float)(RO) / threadsPerBlock.x)));
        unsigned int nbBlocksE1 = (unsigned int )(std::ceil((float)((float)(E1) / threadsPerBlock.y)));

        dim3 numBlocks(nbBlocksRO, nbBlocksE1, 1); 
        
        subsampling_2D<<<numBlocks, threadsPerBlock>>>(data_in, data_out_level_0, data_out_level_1, data_out_level_2, data_out_level_3, RO, E1, SLC);
    }

    template <class T> void 
    execute_subsampling_3D(T *data_in, T *data_out_level_0, T *data_out_level_1, T *data_out_level_2, T *data_out_level_3, int RO, int E1, int SLC)
    {
        dim3 threadsPerBlock(32, 32, 1);

        unsigned int nbBlocksRO = (unsigned int )(std::ceil((float)((float)(RO) / threadsPerBlock.x)));
        unsigned int nbBlocksE1 = (unsigned int )(std::ceil((float)((float)(E1) / threadsPerBlock.y)));

        dim3 numBlocks(nbBlocksRO, nbBlocksE1, SLC); 
        
        subsampling_3D<<<numBlocks, threadsPerBlock>>>(data_in, data_out_level_0, data_out_level_1, data_out_level_2, data_out_level_3, RO, E1, SLC);
    }

    template EXPORTGPUPMRI void execute_subsampling_2D(complext<float> *data_in, complext<float> *data_out_level_0, complext<float> *data_out_level_1, complext<float> *data_out_level_2, complext<float> *data_out_level_3, int RO, int E1, int SLC);
    template EXPORTGPUPMRI void execute_subsampling_3D(complext<float> *data_in, complext<float> *data_out_level_0, complext<float> *data_out_level_1, complext<float> *data_out_level_2, complext<float> *data_out_level_3, int RO, int E1, int SLC);
}
    
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
    __global__ void zero_3D (const T *data_in, T *data_out, int RO, int E1, int SLC, int scaling, int offset_RO, int offset_E1)
    {
        //RO : size of a column
        //E1 : size of a row
        //SLC : current slice
        //scaling : scaling between input and output image
        //offset_X : Offset in rows
        //offset_Y : Offets in columns

        int ro = threadIdx.x + blockIdx.x * blockDim.x; //RO : rowIdx
        int e1 = threadIdx.y + blockIdx.y * blockDim.y; //E1 : colIdx
        int slc = blockIdx.z;                           //Z  : SLC

        int index_slc_in = slc * RO * E1;
        int index_slc_out = slc * RO * scaling * E1 * scaling;

        int index_image_in= ro + RO* e1; // ro : current column
        int index_image_out = (ro + offset_RO) + scaling * RO * (e1 + offset_E1);
        //int index_image_out = ro + (RO * scaling) * e1+ offset_RO + offset_E1 * RO * scaling;
        

        if ( ro < RO && e1 < E1 && slc < SLC )
        {
            data_out[index_image_out + index_slc_out] = data_in[index_image_in + index_slc_in];
        }
    }

    template <class T>
    __global__ void zero_3D_complex (const std::complex<T> *data_in, std::complex<T> *data_out, int RO, int E1, int SLC, int scaling, int offset_RO, int offset_E1)
    {
        //RO : size of a column
        //E1 : size of a row
        //SLC : current slice
        //scaling : scaling between input and output image
        //offset_X : Offset in rows
        //offset_Y : Offets in columns

        int ro = threadIdx.x + blockIdx.x * blockDim.x; //RO : rowIdx
        int e1 = threadIdx.y + blockIdx.y * blockDim.y; //E1 : colIdx
        int slc = blockIdx.z;                           //Z  : SLC

        int index_slc_in = slc * RO * E1;
        int index_slc_out = slc * RO * scaling * E1 * scaling;

        int index_image_in= ro + RO* e1; // ro : current column
        int index_image_out = (ro + offset_RO) + scaling * RO * (e1 + offset_E1);
        //int index_image_out = ro + (RO * scaling) * e1+ offset_RO + offset_E1 * RO * scaling;
        

        if ( ro < RO && e1 < E1 && slc < SLC )
        {
            data_out[index_image_out + index_slc_out] = data_in[index_image_in + index_slc_in];
        }
    }

    template <class T> void 
    execute_zero_3D(T *data_in, T *data_out, int RO, int E1, int SLC, int scaling)
    {
        //To modify eventually
        int nbBlocksSlc = 1 << (int)(log2(SLC));
        if (log2(SLC) != (int)(log2(SLC)))
            nbBlocksSlc = nbBlocksSlc << 1;
    
        //int nbBlocksRO = 1 << (int)(log2(RO) + 1);
        //int nbBlocksE1 = 1 << (int)(log2(E1) + 1);

        dim3 threadsPerBlock(32, 32, 1);
            
        unsigned int nbBlocksRO = (unsigned int )(std::ceil((float)((float)(RO) / threadsPerBlock.x)));
        unsigned int nbBlocksE1 = (unsigned int )(std::ceil((float)((float)(E1) / threadsPerBlock.y)));

        dim3 numBlocks(nbBlocksRO, nbBlocksE1, nbBlocksSlc); 
        
        // std::cout << "scaling: " << scaling << std::endl;
        // std::cout << RO << ", " << E1 << std::endl;
        // std::cout << "Threads per block : ["  << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << "]" << std::endl;
        // std::cout << "Number of blocks : [" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << "]" << std::endl;

        int offsetRO = ((RO * scaling) - RO) / 2;
        int offsetE1 = ((E1 * scaling) - E1) / 2;

        //std::cout << "offsets: " << offsetRO << ", " << offsetE1 << std::endl;

        zero_3D<<<numBlocks, threadsPerBlock>>>(data_in, data_out, RO, E1, SLC, scaling, offsetRO, offsetE1);
    }

    template <class T> void 
    execute_zero_3D_complex(std::complex<T> *data_in, std::complex<T> *data_out, int RO, int E1, int SLC, int scaling)
    {
        //To modify eventually
        int nbBlocksSlc = 1 << (int)(log2(SLC));
        if (log2(SLC) != (int)(log2(SLC)))
            nbBlocksSlc = nbBlocksSlc << 1;
    
        std::cout << "nb blocks slices: " << nbBlocksSlc << std::endl;
        std::cout << "nb slices: " << SLC << std::endl;

        dim3 threadsPerBlock(32, 32, 1);
            
        unsigned int nbBlocksRO = (unsigned int )(std::ceil((float)((float)(RO) / threadsPerBlock.x)));
        unsigned int nbBlocksE1 = (unsigned int )(std::ceil((float)((float)(E1) / threadsPerBlock.y)));

        dim3 numBlocks(nbBlocksRO, nbBlocksE1, nbBlocksSlc); 
        
        // std::cout << "scaling: " << scaling << std::endl;
        // std::cout << RO << ", " << E1 << std::endl;
        // std::cout << "Threads per block : ["  << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << "]" << std::endl;
        // std::cout << "Number of blocks : [" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << "]" << std::endl;

        int offsetRO = ((RO * scaling) - RO) / 2;
        int offsetE1 = ((E1 * scaling) - E1) / 2;

        //std::cout << "offsets: " << offsetRO << ", " << offsetE1 << std::endl;

        zero_3D_complex<<<numBlocks, threadsPerBlock>>>(data_in, data_out, RO, E1, SLC, scaling, offsetRO, offsetE1);
    }

    template EXPORTGPUPMRI void execute_zero_3D(float *data_in, float *data_out, int RO, int E1, int SLC, int scaling);
    template EXPORTGPUPMRI void execute_zero_3D_complex(std::complex<float> *data_in, std::complex<float> *data_out, int RO, int E1, int SLC, int scaling);
}
    
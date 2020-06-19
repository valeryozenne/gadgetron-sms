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
using namespace std;

namespace Gadgetron{

    template<class REAL> 
    __global__ void copy_cuNDArray_benoit(complext<REAL> *in_data, complext<REAL> *out_data, int *dimensions)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < dimensions[0])
        {
            out_data[i] = in_data[i];
        }
    }

    template<class REAL> 
    void create_and_copy_cuNDArray_benoit( cuNDArray<complext<REAL> > data_in, cuNDArray<complext<REAL> > data_out)
    {

        int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
        int warp_size = cudaDeviceManager::Instance()->warp_size(cur_device);//thread size
        int max_blockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);//block size
        //dim3 blockDim(((max_blockdim/CHA)/warp_size)*warp_size, CHA);
        
        //GDEBUG_STREAM("warp size: " << warp_size << ", blockDim = " << max_blockDim << std::endl);
        int max_thread_x_size = 1024;
        int nbBlocks = (data_in.dimensions()[0] * data_in.dimensions()[1]) / max_thread_x_size;

        int *dimensions = new int(data_in.dimensions().size());
        for (unsigned int i = 0; i < data_in.dimensions().size(); i++)
        {
            dimensions[i] = data_in.dimensions()[i];
        }

        // out_gpu_data.create(data_in.get_dimensions());
        //copy_cuNDArray_benoit<<<nbBlocks, max_thread_x_size>>>(data_in.get_data_ptr(), data_out.get_data_ptr(), dimensions);
        // CHECK_FOR_CUDA_ERROR();
    }

    /*void create_and_copy_benoit_2(hondarray<complext<REAL> >data_in)
    {

    }*/

    //template EXPORTGPUPMRI __global__ void copy_cuNDArray_benoit( cuNDArray<complext<float> > , cuNDArray<complext<float> >, int *dimensions);
    template EXPORTGPUPMRI void create_and_copy_cuNDArray_benoit(cuNDArray<complext<float> > data_in, cuNDArray<complext<float> > data_out);
}


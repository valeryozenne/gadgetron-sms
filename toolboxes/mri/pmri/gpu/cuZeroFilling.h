#pragma once

#include "gpupmri_export.h"
#include "cuNDArray.h"
#include "vector_td.h"
#include "complext.h"

#include <boost/shared_ptr.hpp>

namespace Gadgetron
{

    //template<class REAL> 
    //global__ void copy_cuNDArray_benoit(cuNDArray< complext<REAL> > in_data, cuNDArray< complext<REAL> > out_data);
    //template<class REAL> 
    void execute_zerofilling_gpu(complext<float> *data_in, complext<float> *data_out, int *in_dimensions, int *out_dimensions, int offsetX, int offsetY, size_t sizeX, size_t sizeY);
}
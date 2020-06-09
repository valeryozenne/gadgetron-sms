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
    template<class REAL> 
    void create_and_copy_cuNDArray_benoit( cuNDArray<complext<REAL> > data_in, cuNDArray<complext<REAL> > data_out);
}
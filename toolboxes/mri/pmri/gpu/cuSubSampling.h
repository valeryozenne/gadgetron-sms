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
    template<class REAL> EXPORTGPUPMRI void
    execute_subsampling_2D(REAL *data_in, REAL *data_out_level_0, REAL *data_out_level_1, REAL *data_out_level_2, REAL *data_out_level_3, int RO, int E1, int SLC);
    template<class REAL> EXPORTGPUPMRI void
    execute_subsampling_3D(REAL *data_in, REAL *data_out_level_0, REAL *data_out_level_1, REAL *data_out_level_2, REAL *data_out_level_3, int RO, int E1, int SLC);
}

/** \file   mri_core_grappa.h
    \brief  GRAPPA implementation for 2D and 3D MRI parallel imaging
    \author Hui Xue
*/

#pragma once

#include "mri_core_export.h"
#include "hoNDArray.h"
#include "hoArmadillo.h"

namespace Gadgetron {


 template <typename T> EXPORTMRICORE void im2col(hoNDArray< T >& input, hoNDArray<T >& block_SB, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1);
 template <typename T> EXPORTMRICORE void remove_unnecessary_kspace(hoNDArray<T>& input, hoNDArray<T>& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );
 template <typename T> EXPORTMRICORE void extract_milieu_kernel(hoNDArray< T >& block_SB, hoNDArray< T >& missing_data, const size_t kernel_size, const size_t voxels_number_per_image);
 template <typename T> EXPORTMRICORE void apply_unmix_coeff_kspace_SMS(hoNDArray<T>& in, hoNDArray<T>& kernel, hoNDArray<T>& out);
}

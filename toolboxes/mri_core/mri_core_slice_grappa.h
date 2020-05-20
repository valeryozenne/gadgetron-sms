
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
 template <typename T> EXPORTMRICORE void im2col_open(hoNDArray< T >& input, hoNDArray<T >& block_SB, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1);
 template <typename T> EXPORTMRICORE void remove_unnecessary_kspace(hoNDArray<T>& input, hoNDArray<T>& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );
 template <typename T> EXPORTMRICORE void remove_unnecessary_kspace_open(hoNDArray<T>& input, hoNDArray<T>& output, const size_t acc , const size_t startE1, const size_t endE1, bool is_mb );
 template <typename T> EXPORTMRICORE void extract_milieu_kernel(hoNDArray< T >& block_SB, hoNDArray< T >& missing_data, const size_t kernel_size, const size_t voxels_number_per_image);
 template <typename T> EXPORTMRICORE void apply_unmix_coeff_kspace_SMS(hoNDArray<T>& in, hoNDArray<T>& kernel, hoNDArray<T>& out);


template <typename T> EXPORTMRICORE void create_stacks_of_slices_directly_sb(hoNDArray< T >& data, hoNDArray< T >& new_stack , std::vector<unsigned int> &indice, std::vector< std::vector<unsigned int> > &MapSliceSMS);
template <typename T> EXPORTMRICORE void create_stacks_of_slices_directly_sb_open(hoNDArray< T >& data, hoNDArray< T >& new_stack , std::vector<unsigned int> &indice, std::vector< std::vector<unsigned int> > &MapSliceSMS);

template <typename T> EXPORTMRICORE void create_stacks_of_slices_directly_mb(hoNDArray<T >& mb,hoNDArray< T>& mb_8D , std::vector<unsigned int>& indice_mb, std::vector<unsigned int>& indice_slice_mb );
template <typename T> EXPORTMRICORE void create_stacks_of_slices_directly_mb_open(hoNDArray<T >& mb,hoNDArray< T>& mb_8D , std::vector<unsigned int>& indice_mb, std::vector<unsigned int>& indice_slice_mb );

template <typename T> EXPORTMRICORE void undo_stacks_ordering_to_match_gt_organisation(hoNDArray< T >& data, hoNDArray< T > &output, std::vector< std::vector<unsigned int> >& MapSliceSMS, std::vector<unsigned int>& indice_sb);
template <typename T> EXPORTMRICORE void undo_stacks_ordering_to_match_gt_organisation_open(hoNDArray< T >& data, hoNDArray< T > &output, std::vector< std::vector<unsigned int> >& MapSliceSMS, std::vector<unsigned int>& indice_sb);

}

#pragma once

#include "hoCuNDArray.h"
#include "cuNDArray.h"
#include "vector_td.h"

namespace Gadgetron {
  
  // Forwards projection of a 3D volume onto a set of projections.
  // - dependening on the provided binnning indices, just a subset of the projections can be targeted.
  //
  
  void 
  conebeam_forwards_projection( 
        hoCuNDArray<float> *projections,
				hoCuNDArray<float> *image,
				std::vector<float> angles, 
				std::vector<floatd2> offsets, 
				std::vector<unsigned int> indices,
				int projections_per_batch, 
				float samples_per_pixel,
				floatd3 is_dims_in_mm, 
				floatd2 ps_dims_in_mm,
				float SDD, 
				float SAD,
				bool accumulate 
  );
  
  // Backprojection of a set of projections onto a 3D volume.
  // - depending on the provided binnning indices, just a subset of the projections can be included
  //

  template <bool FBP>
  void conebeam_backwards_projection( 
        hoCuNDArray<float> *projections,
        hoCuNDArray<float> *image,
        std::vector<float> angles, 
        std::vector<floatd2> offsets, 
        std::vector<unsigned int> indices,
        int projections_per_batch,
        intd3 is_dims_in_pixels, 
        floatd3 is_dims_in_mm, 
        floatd2 ps_dims_in_mm,
        float SDD, 
        float SAD,
        bool accumulate, 
        bool short_scan,
        bool use_offset_correction,
        cuNDArray<float> *cosine_weights = 0x0,
        cuNDArray<float> *frequency_filter = 0x0
  );
}

#include "hoNDArray_fileio.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_elemwise.h"
#include "parameterparser.h"
#include "cuConvolutionOperator.h"
#include <iostream>
#include <math.h>
#include "cuSubSampling.h"
#include "ImageIOAnalyze.h"

using namespace Gadgetron;


    int main()
    {
        ImageIOAnalyze gt_exporter;

        hoNDArray<std::complex<float> > in(4, 4, 4);
        
        gt_exporter.import_array_complex(in, "/tmp/gadgetron/zerofilling_standalone_entree_gpu_REAL", "/tmp/gadgetron/zerofilling_standalone_entree_gpu_IMAG");

        std::vector<hoNDArray<std::complex<float> >> out(4, hoNDArray<std::complex<float> >());
        out[0].create(in.get_size(0), in.get_size(1));
        out[1].create(in.get_size(0) / 2, in.get_size(1) / 2);
        out[2].create(in.get_size(0) / 4, in.get_size(1) / 4);
        out[3].create(in.get_size(0) / 8, in.get_size(1) / 8);

            //in.get_size(0), in.get_size(1), in.get_size(6));

        size_t RO = in.get_size(0);
        size_t E1 = in.get_size(1);
        size_t E2 = in.get_size(2);
        size_t CHA = in.get_size(3);
        size_t N = in.get_size(4);
        size_t S = in.get_size(5);
        size_t SLC = in.get_size(6);

        cuNDArray<complext<float> >dev_in(RO, E1, E2, CHA, N, S, SLC);
        
        std::complex<float> *pIn = &(in(0, 0, 0, 0, 0, 0, 0));
        if (cudaMemcpy(dev_in.get_data_ptr(), pIn, RO * E1 * E2 * CHA * N * S * SLC * sizeof(std::complex<float>), cudaMemcpyHostToDevice) != cudaSuccess)
            GERROR("Upload to device from in_data failed\n");
        //cuNDArray<std::complex<float> >dev_out(out);
        std::vector<cuNDArray<complext<float> > >dev_out(4);
        dev_out[0].create(in.get_size(0)    , in.get_size(1)    , 1, CHA, N, S, SLC);
        dev_out[1].create(in.get_size(0) / 2, in.get_size(1) / 2, 1, CHA, N, S, SLC);
        dev_out[2].create(in.get_size(0) / 4, in.get_size(1) / 4, 1, CHA, N, S, SLC);
        dev_out[3].create(in.get_size(0) / 8, in.get_size(1) / 8, 1, CHA, N, S, SLC);
        
        execute_subsampling_3D(dev_in.get_data_ptr(), dev_out[0].get_data_ptr(), dev_out[1].get_data_ptr(), dev_out[2].get_data_ptr(), dev_out[3].get_data_ptr(), in.get_size(0), in.get_size(1), in.get_size(6));

        auto output_ptr = dev_out[0].to_host();
        out[0] =  std::move(reinterpret_cast<hoNDArray<std::complex<float>>&>(*output_ptr));
        output_ptr = dev_out[1].to_host();
        out[1] =  std::move(reinterpret_cast<hoNDArray<std::complex<float>>&>(*output_ptr));
        output_ptr = dev_out[2].to_host();
        out[2] =  std::move(reinterpret_cast<hoNDArray<std::complex<float>>&>(*output_ptr));
        output_ptr = dev_out[3].to_host();
        out[3] =  std::move(reinterpret_cast<hoNDArray<std::complex<float>>&>(*output_ptr));

        
        gt_exporter.export_array_complex(in, "/tmp/gadgetron/subsampling_standalone_entree_gpu");
        gt_exporter.export_array_complex(out[0], "/tmp/gadgetron/subsampling_standalone_sortie_gpu_level_0");
        gt_exporter.export_array_complex(out[1], "/tmp/gadgetron/subsampling_standalone_sortie_gpu_level_1");
        gt_exporter.export_array_complex(out[2], "/tmp/gadgetron/subsampling_standalone_sortie_gpu_level_2");
        gt_exporter.export_array_complex(out[3], "/tmp/gadgetron/subsampling_standalone_sortie_gpu_level_3");

        return(0);
    }
    
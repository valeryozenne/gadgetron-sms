#include "hoNDArray_fileio.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_elemwise.h"
#include "parameterparser.h"
#include "cuConvolutionOperator.h"
#include <iostream>
#include <math.h>
#include "cuZeroFilling.h"
#include "ImageIOAnalyze.h"

using namespace Gadgetron;


    int main()
    {
        ImageIOAnalyze gt_exporter;

        hoNDArray<std::complex<float> > in(4, 4, 4);
        
        gt_exporter.import_array_complex(in, "/tmp/gadgetron/outBeforeZeroFilling_REAL", "/tmp/gadgetron/outBeforeZeroFilling_IMAG");

        int scaling = 2;

        hoNDArray<std::complex<float> > out(in.get_size(0) * scaling, in.get_size(1) * scaling, in.get_size(6));

        
        //in.fill(255);
        out.fill(0);

        cuNDArray<std::complex<float> >dev_in(in);
        cuNDArray<std::complex<float> >dev_out(out);

        // for (int i = 0; i < in.get_number_of_elements(); i++)
        // {
        //     std::cout << dev_in[i] << ", ";
        // }
        // std::cout << std::endl;

        execute_zero_3D_complex(dev_in.get_data_ptr(), dev_out.get_data_ptr(), in.get_size(0), in.get_size(1), in.get_size(6), scaling);

        auto output_ptr = dev_out.to_host();
        out =  std::move(reinterpret_cast<hoNDArray<std::complex<float>>&>(*output_ptr));

        
        gt_exporter.export_array_complex(in, "/tmp/gadgetron/zerofilling_standalone_entree_gpu");
        gt_exporter.export_array_complex(out, "/tmp/gadgetron/zerofilling_standalone_sortie_gpu");

        // for (int i = 0; i < out.get_number_of_elements(); i++)
        // {
        //     std::cout << out[i] << ", ";
        // }
        // std::cout << std::endl;

        return(0);
    }
    
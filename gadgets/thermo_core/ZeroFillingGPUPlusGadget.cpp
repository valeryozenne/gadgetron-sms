
#include "ZeroFillingGPUPlusGadget.h"
#include <iomanip>
#include <sstream>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "mri_core_utility.h"

#include "cuNDArray.h"
#include "cuNDFFT.h"
#include "cuZeroFilling.h"
//tmp include to check cuda issues
#include <helper_cuda.h>

namespace Gadgetron {

ZeroFillingGPUPlusGadget::ZeroFillingGPUPlusGadget() : BaseClass()
{
}

ZeroFillingGPUPlusGadget::~ZeroFillingGPUPlusGadget()
{
}

int ZeroFillingGPUPlusGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    ISMRMRD::IsmrmrdHeader h;
    try
    {
        deserialize(mb->rd_ptr(), h);
    }
    catch (...)
    {
        GDEBUG("Error parsing ISMRMRD Header");
    }

    if (!h.acquisitionSystemInformation)
    {
        GDEBUG("acquisitionSystemInformation not found in header. Bailing out");
        return GADGET_FAIL;
    }



    // -------------------------------------------------

    return GADGET_OK;
}

int ZeroFillingGPUPlusGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
{
    if (perform_timing.value()) { gt_timer_local_.start("ZeroFillingGPUPlusGadget::process"); }

    GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGPUPlusGadget::process(...) starts ... ");

    // -------------------------------------------------------------

    process_called_times_++;

    // -------------------------------------------------------------

    IsmrmrdImageArray* data = m1->getObjectPtr();

    Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>* cm2 = new Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>();

    IsmrmrdImageArray map_sd;

    GDEBUG_STREAM("Oversampling before zero_filling: " << oversampling << std::endl);
    // if (use_gpu.value())
    // {
          perform_zerofilling_array_gpu(*data,  map_sd);
    // }
    // else
    // {
    //    perform_zerofilling_array(*data,  map_sd);
    // }



    // print out data info
    if (verbose.value())
    {
        GDEBUG_STREAM("----> ZeroFillingGPUPlusGadget::process(...) has been called " << process_called_times_ << " times ...");
        std::stringstream os;
        data->data_.print(os);
        GDEBUG_STREAM(os.str());

        std::stringstream os2;
        map_sd.data_.print(os2);
        GDEBUG_STREAM(os2.str());
    }

    *(cm2->getObjectPtr()) = map_sd;
    if (this->next()->putq(cm2) == -1)
    {
        GERROR("CmrParametricMappingGadget::process, passing sd map on to next gadget");
        return GADGET_FAIL;
    }

    m1->release();


    //put data on the gpu

    //perform zerofilling

    // extract partie reelle
    // extract partie imaginaire
    // perform motion correction
    // rassemble les données

    // -------------------------------------------------------------


    if (perform_timing.value()) { gt_timer_local_.stop(); }

    return GADGET_OK;
}

void ZeroFillingGPUPlusGadget::perform_zerofilling_array(IsmrmrdImageArray& in, IsmrmrdImageArray& out)
{

    size_t RO = in.data_.get_size(0);
    size_t E1 = in.data_.get_size(1);
    size_t E2 = in.data_.get_size(2);
    size_t CHA = in.data_.get_size(3);
    size_t N = in.data_.get_size(4);
    size_t S = in.data_.get_size(5);
    size_t SLC = in.data_.get_size(6);

    size_t rep   = in.headers_(0, 0, 0).repetition;



        out.data_.create(RO*oversampling.value(), E1*oversampling.value(), E2, CHA, N, S, SLC);
        out.headers_.create(N, S, SLC);
        out.meta_.resize(N*S*SLC);


    Gadgetron::clear(out.data_);

    
    perform_zerofilling(in.data_, out.data_);

    size_t n, s, slc;

    for (slc = 0; slc < SLC; slc++)
    {
        for (s = 0; s < S; s++)
        {
            for (n = 0; n < N; n++)
            {
                out.headers_(n, s, slc) = in.headers_(n, s, slc);

                out.headers_(n, s, slc).matrix_size[0]=in.headers_(n, s, slc).matrix_size[0]*oversampling.value();
                out.headers_(n, s, slc).matrix_size[1]=in.headers_(n, s, slc).matrix_size[1]*oversampling.value();


            }
        }
    }
}

void ZeroFillingGPUPlusGadget::perform_zerofilling(hoNDArray< std::complex<float> > & data_in, hoNDArray< std::complex<float> > & data_out)
{
    try
    {
        if (perform_timing.value()) { gt_timer_.start("ZeroFillingGPUPlusGadget::perform_filling"); }

        GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGPUPlusGadget::perform_zerofilling(...) starts ... ");


        if (oversampling.value()>1)
        {

            size_t RO = data_in.get_size(0);
            size_t E1 = data_in.get_size(1);
            size_t E2 = data_in.get_size(2);
            size_t CHA = data_in.get_size(3);
            size_t N = data_in.get_size(4);
            size_t S = data_in.get_size(5);
            size_t SLC = data_in.get_size(6);

            Gadgetron::hoNDFFT<float>::instance()->fft2c(data_in);

            unsigned int offset_ro=int(RO/oversampling.value());
            unsigned int offset_e1=int(E1/oversampling.value());
            GDEBUG("offset %d %d", offset_ro, offset_e1);

            for (long long slc = 0; slc < SLC; slc++)
            {
                for (long long s = 0; s < S; s++)
                {
                    for (long long n = 0; n < N; n++)
                    {
                        for (long long cha = 0; cha < CHA; cha++)
                        {
                            for (long long e2 = 0; e2 < E2; e2++)
                            {
                                for (long long e1 = 0; e1 < E1; e1++)
                                {
                                    std::complex<float> * in = &(data_in(0, e1, e2, cha, n, s, slc));
                                    std::complex<float> * out = &(data_out(offset_ro, e1+offset_e1, e2, cha, n, s, slc));
                                    memcpy(out , in, sizeof(std::complex<float>)*RO);

                                }
                            }
                        }
                    }
                }
            }

            Gadgetron::hoNDFFT<float>::instance()->ifft2c(data_out);
        }
        else
        {
            //TODO we should not pass here if oversampling is 0 , it is only usefull for debug
            memcpy(data_out.begin() , data_in.begin(), sizeof(std::complex<float>)*data_in.get_number_of_elements());
        }
        // -------------------------------------------------------------

        if (perform_timing.value()) { gt_timer_.stop(); }
    }
    catch (...)
    {
        GERROR_STREAM("Exceptions happened in ZeroFillingGPUPlusGadget::perform_zerofilling(...) ... ");
    }


}

void ZeroFillingGPUPlusGadget::perform_zerofilling_array_gpu(IsmrmrdImageArray& in, IsmrmrdImageArray& out)
{

    GDEBUG("----- go GPU -----------\n");

    size_t RO = in.data_.get_size(0);
    size_t E1 = in.data_.get_size(1);
    size_t E2 = in.data_.get_size(2);
    size_t CHA = in.data_.get_size(3);
    size_t N = in.data_.get_size(4);
    size_t S = in.data_.get_size(5);
    size_t SLC = in.data_.get_size(6);

    GDEBUG("Oversampling: %d\n", oversampling.value());
    

    //TODO need to be done once (if REP==0)
    out.data_.create(RO*oversampling.value(), E1*oversampling.value(), E2, CHA, N, S, SLC);
    out.headers_.create(N, S, SLC);
    out.meta_.resize(N*S*SLC);
    
    Gadgetron::clear(out.data_);

    //out.data_.fill(10);

    hoNDArray< complext<float> > data_in(reinterpret_cast<hoNDArray<complext<float>>&>(in.data_));
    hoNDArray< complext<float> > data_out(reinterpret_cast<hoNDArray<complext<float>>&>(out.data_));

    ///////////////////////////////////////////////////////////////
    //tmp allocation - to change

    ///////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////
    //Correct allocation

    cu_data.create(RO, E1, E2, CHA, N, S, SLC);
    cu_data_out.create(RO * oversampling, E1 * oversampling, E2, CHA, N, S, SLC);

    // complext<float> *in_ptr, *out_ptr, *cu_in_ptr, *cu_out_ptr;
    // in_ptr = data_in.get_data_ptr();
    // out_ptr = data_out.get_data_ptr();
    // cu_in_ptr = cu_data.get_data_ptr();
    // cu_out_ptr = cu_data_out.get_data_ptr();
    
    // if (cudaMalloc((void **)(&(cu_in_ptr)), sizeof(complext<float>) * RO * E1 * E2 * CHA * N * S * SLC) != cudaSuccess)
    // {
    //     GERROR("CANNOT ALLOCATE CU_IN_PTR\n");
    // }
    // if (cudaMalloc((void **)(&(cu_out_ptr)), sizeof(complext<float>) * RO * oversampling.value() * E1 * oversampling.value() * E2 * CHA * N * S * SLC) != cudaSuccess)
    // {
    //     GERROR("CANNOT ALLOCATE CU_OUT_PTR\n");
    // }
    std::complex<float> *pIn = &(in.data_(0, 0, 0, 0, 0, 0, 0));
    if (cudaMemcpy(cu_data.get_data_ptr(), pIn, RO * E1 * E2 * CHA * N * S * SLC * sizeof(std::complex<float>), cudaMemcpyHostToDevice) != cudaSuccess)
        GERROR("Upload to device from in_data failed\n");

    // ///////////////////////////////////////////////////////////////
    
    cuNDFFT<float>::instance()->fft2(&cu_data);

    execute_zero_3D_complext(cu_data.get_data_ptr(), cu_data_out.get_data_ptr(), in.data_.get_size(0), in.data_.get_size(1), in.data_.get_size(6), oversampling);
    
    cuNDFFT<float>::instance()->ifft2(&cu_data_out);
    
    ///////////////////////////////////////////////////////////////////////////////
    //Code for solution 1 - WORKS
    //auto output_ptr = cu_data_out.to_host();
    //out.data_ =  std::move(reinterpret_cast<hoNDArray<std::complex<float>>&>(*output_ptr));

    //code for solution 2 - WORKS BUT ERRORS WITH FFT
    std::complex<float> *pOut = &(out.data_(0, 0, 0, 0, 0, 0, 0));
    if (cudaMemcpy(pOut, cu_data_out.get_data_ptr(), RO * oversampling * E1 * oversampling * E2 * CHA * N * S * SLC * sizeof(std::complex<float>), cudaMemcpyDeviceToHost) != cudaSuccess)
        GERROR("Upload to device from in_data failed\n");
    ///////////////////////////////////////////////////////////////////////////////
    // GDEBUG_STREAM("data_out memory size: " << data_out.get_number_of_bytes() << " bytes");
    // GDEBUG_STREAM("copy size: " << RO * oversampling.value() * E1 * oversampling.value() * E2 * CHA * N * S * SLC * sizeof(complext<float>) << " bytes");
    // GDEBUG_STREAM("cu_data_out size: " << cu_data_out.get_number_of_bytes() << " bytes");
    // int error = cudaGetLastError();
    // GDEBUG_STREAM("Last CUDA error before memcpy: " << error);
    // if (cudaMemcpy(out_ptr, cu_out_ptr, RO * oversampling.value() * E1 * oversampling.value() * E2 * CHA * N * S * SLC * sizeof(complext<float>), cudaMemcpyDeviceToHost) != cudaSuccess)
    //     GERROR("Upload to host from cu_data_out failed\n");
        
        
        
    ///////////////////////////////////////////////////////////////////////////////

    std::string outfile = "/tmp/gadgetron/zerofilling_standalone_sortie_gpu";
    std::string infile = "/tmp/gadgetron/zerofilling_standalone_entree_gpu";
    gt_exporter_.export_array_complex(in.data_, infile);
    gt_exporter_.export_array_complex(out.data_, outfile);

    size_t n, s, slc;

    for (slc = 0; slc < SLC; slc++)
    {
        for (s = 0; s < S; s++)
        {
            for (n = 0; n < N; n++)
            {
                out.headers_(n, s, slc) = in.headers_(n, s, slc);

                out.headers_(n, s, slc).matrix_size[0]=in.headers_(n, s, slc).matrix_size[0]*oversampling.value();
                out.headers_(n, s, slc).matrix_size[1]=in.headers_(n, s, slc).matrix_size[1]*oversampling.value();


            }
        }
    }
    CHECK_FOR_CUDA_ERROR();
}

// ----------------------------------------------------------------------------------------

GADGET_FACTORY_DECLARE(ZeroFillingGPUPlusGadget)

}

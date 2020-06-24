
#include "ZeroFillingGPUGadget.h"
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

namespace Gadgetron {

ZeroFillingGPUGadget::ZeroFillingGPUGadget() : BaseClass()
{
}

ZeroFillingGPUGadget::~ZeroFillingGPUGadget()
{
}

int ZeroFillingGPUGadget::process_config(ACE_Message_Block* mb)
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

int ZeroFillingGPUGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
{
    if (perform_timing.value()) { gt_timer_local_.start("ZeroFillingGPUGadget::process"); }

    GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGPUGadget::process(...) starts ... ");

    // -------------------------------------------------------------

    process_called_times_++;

    // -------------------------------------------------------------

    IsmrmrdImageArray* data = m1->getObjectPtr();

    Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>* cm2 = new Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>();

    IsmrmrdImageArray map_sd;

    GDEBUG_STREAM("Oversampling before zero_filling: " << oversampling << std::endl);
    if (use_gpu.value())
    {
        perform_zerofilling_array_gpu(*data,  map_sd);
    }
    else
    {
        perform_zerofilling_array(*data,  map_sd);
    }



    // print out data info
    if (verbose.value())
    {
        GDEBUG_STREAM("----> ZeroFillingGPUGadget::process(...) has been called " << process_called_times_ << " times ...");
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


void ZeroFillingGPUGadget::perform_zerofilling_array_gpu(IsmrmrdImageArray& in, IsmrmrdImageArray& out)
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

    out.data_.fill(10);


    cuNDArray< complext<float> > cu_data_out(reinterpret_cast<const hoNDArray<complext<float>> &>(out.data_));
    cuNDArray< complext<float> > cu_data(reinterpret_cast<const hoNDArray<complext<float>> &>(in.data_));

    //cuNDFFT<float>::instance()->fft2d(&cu_data);

    GDEBUG("-----------------------------------BEFORE PERFORM_ZEROFILLING-----------------------------------\n");
    perform_zerofilling_gpu(cu_data, cu_data_out);
    GDEBUG("-----------------------------------PERFORM_ZEROFILLING DONE-----------------------------------\n");
    
    //cuNDFFT<float>::instance()->ifft2(&cu_data_out);

    //perform_zerofilling(in.data_, out.data_);

    //cu_data_out.to_host(reinterpret_cast<const hoNDArray<complext<float>> >(out.data_));
    

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

void ZeroFillingGPUGadget::perform_zerofilling_array(IsmrmrdImageArray& in, IsmrmrdImageArray& out)
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





void ZeroFillingGPUGadget::perform_zerofilling(hoNDArray< std::complex<float> > & data_in, hoNDArray< std::complex<float> > & data_out)
{
    try
    {
        if (perform_timing.value()) { gt_timer_.start("ZeroFillingGPUGadget::perform_filling"); }

        GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGPUGadget::perform_zerofilling(...) starts ... ");


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
        GERROR_STREAM("Exceptions happened in ZeroFillingGPUGadget::perform_zerofilling(...) ... ");
    }


}

void ZeroFillingGPUGadget::perform_zerofilling_gpu(cuNDArray<complext<float>> data_in, cuNDArray<complext<float>> data_out)
{
    //try
    //{
        if (perform_timing.value()) { gt_timer_.start("ZeroFillingGPUGadget::perform_filling"); }

        GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGPUGadget::perform_zerofilling(...) starts ... ");
        GDEBUG("data_in dimensions: %d, %d, %d, %d, %d, %d, %d\n", data_in.dimensions()[0], data_in.dimensions()[1], data_in.dimensions()[2], data_in.dimensions()[3], data_in.dimensions()[4], data_in.dimensions()[5], data_in.dimensions()[6]);
        GDEBUG("data_out dimensions: %d, %d, %d, %d, %d, %d, %d\n", data_out.dimensions()[0], data_out.dimensions()[1], data_out.dimensions()[2], data_out.dimensions()[3], data_out.dimensions()[4], data_out.dimensions()[5], data_out.dimensions()[6]);

        
        if (oversampling.value()>1)
        {

            size_t RO = data_in.get_size(0);
            size_t E1 = data_in.get_size(1);
            size_t E2 = data_in.get_size(2);
            size_t CHA = data_in.get_size(3);
            size_t N = data_in.get_size(4);
            size_t S = data_in.get_size(5);
            size_t SLC = data_in.get_size(6);

            //Gadgetron::hoNDFFT<float>::instance()->fft2c(data_in);

            unsigned int offset_ro=int(RO/oversampling.value());
            unsigned int offset_e1=int(E1/oversampling.value());
            GDEBUG("offset %d %d\n", offset_ro, offset_e1);

            int *in_data_dimensions = new int[data_in.get_number_of_dimensions()];
            GDEBUG_STREAM("NUMBER OF DIMENSIONS: " << data_in.get_number_of_dimensions());
            for (unsigned int i = 0; i < data_in.get_number_of_dimensions(); i++)
            {
                in_data_dimensions[i] = data_in.get_size(i);
                GDEBUG_STREAM("Dimension " << i << ": " << in_data_dimensions[i]);
            }

            int *out_data_dimensions = new int[data_out.get_number_of_dimensions()];
            for (unsigned int i = 0; i < data_out.get_number_of_dimensions(); i++)
            {
                out_data_dimensions[i] = data_out.get_size(i);
            }

            
            //first complext array: data_in.get_data_ptr()
            //second complext array: data_out.get_data_ptr()

            //global function call arguments : data_in, data_out, dimensions, offsets
            //GDEBUG("-------------------------BEFORE EXECUTE_ZEROFILLING ------------------------------\n");
            execute_zerofilling_gpu(data_in.get_data_ptr(), data_out.get_data_ptr(), in_data_dimensions, out_data_dimensions, offset_ro, offset_e1, RO, E1);
            //GDEBUG("-------------------------AFTER EXECUTE_ZEROFILLING ------------------------------\n");
            //*******UNUSED CODE***************//
            // for (long long slc = 0; slc < SLC; slc++)
            // {
            //     for (long long s = 0; s < S; s++)
            //     {
            //         for (long long n = 0; n < N; n++)
            //         {
            //             for (long long cha = 0; cha < CHA; cha++)
            //             {
            //                 for (long long e2 = 0; e2 < E2; e2++)
            //                 {
            //                     for (long long e1 = 0; e1 < E1; e1++)
            //                     {
            //                         std::complex<float> * in = &(data_in(0, e1, e2, cha, n, s, slc));
            //                         std::complex<float> * out = &(data_out(offset_ro, e1+offset_e1, e2, cha, n, s, slc));
            //                         memcpy(out , in, sizeof(std::complex<float>)*RO);

            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }

            //Gadgetron::hoNDFFT<float>::instance()->ifft2c(data_out);
        }
        else
        {
            //TODO we should not pass here if oversampling is 0 , it is only usefull for debug
            //memcpy(data_out.begin() , data_in.begin(), sizeof(std::complex<float>)*data_in.get_number_of_elements());
        }

        // -------------------------------------------------------------

        if (perform_timing.value()) { gt_timer_.stop(); }
    /*}
    catch (...)
    {
        GERROR_STREAM("Exceptions happened in ZeroFillingGPUGadget::perform_zerofilling(...) ... ");
    }*/
    GDEBUG("PERFORM_ZEROFILLING_GPU DONE\n");
}



// ----------------------------------------------------------------------------------------

GADGET_FACTORY_DECLARE(ZeroFillingGPUGadget)

}

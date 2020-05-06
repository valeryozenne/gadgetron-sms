
#include "ZeroFillingGadget.h"
#include <iomanip>
#include <sstream>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "mri_core_utility.h"


namespace Gadgetron {

ZeroFillingGadget::ZeroFillingGadget() : BaseClass()
{
}

ZeroFillingGadget::~ZeroFillingGadget()
{
}

int ZeroFillingGadget::process_config(ACE_Message_Block* mb)
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

int ZeroFillingGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
{
    if (perform_timing.value()) { gt_timer_local_.start("ZeroFillingGadget::process"); }

    GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGadget::process(...) starts ... ");

    // -------------------------------------------------------------

    process_called_times_++;

    // -------------------------------------------------------------

    unsigned int scaling=2;

    IsmrmrdImageArray* data = m1->getObjectPtr();

    hoNDArray< ISMRMRD::ImageHeader > headers_=m1->getObjectPtr()->headers_;

    size_t RO = data->data_.get_size(0);
    size_t E1 = data->data_.get_size(1);
    size_t E2 = data->data_.get_size(2);
    size_t CHA = data->data_.get_size(3);
    size_t N = data->data_.get_size(4);
    size_t S = data->data_.get_size(5);
    size_t SLC = data->data_.get_size(6);

    size_t hN = headers_.get_size(0);
    size_t hS = headers_.get_size(1);
    size_t hSLC = headers_.get_size(2);

    GDEBUG_STREAM(" N :  "  << hN  <<" S: "<< hS <<" SLC: " << hSLC);

    hoNDArray< std::complex<float> > data_out(RO*scaling, E1*scaling , E2, CHA, N, S, SLC);

    perform_zerofilling(*data, data_out, scaling);


    // print out data info
    if (verbose.value())
    {
        GDEBUG_STREAM("----> ZeroFillingGadget::process(...) has been called " << process_called_times_ << " times ...");
        std::stringstream os;
        data->data_.print(os);
        GDEBUG_STREAM(os.str());

        std::stringstream os2;
        data_out.print(os2);
        GDEBUG_STREAM(os2.str());
    }

    IsmrmrdImageArray map_sd;



    map_sd.data_.create(RO*scaling, E1*scaling , E2, CHA, N, S, SLC);
    Gadgetron::clear(map_sd.data_);
    map_sd.data_=data_out;
    map_sd.headers_.create(N, S, SLC);

    for (long long slc = 0; slc < hSLC; slc++)
    {
        for (long long s = 0; s < hS; s++)
        {
            for (long long n = 0; n < hN; n++)
            {

                map_sd.headers_(0, s, slc) = headers_(0, s, slc);
                map_sd.headers_(n, s, slc).matrix_size[0]=headers_(n, s, slc).matrix_size[0]*scaling;
                map_sd.headers_(n, s, slc).matrix_size[1]=headers_(n, s, slc).matrix_size[1]*scaling;

            }
        }
    }

    Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>* cm2 = new Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>();
    *(cm2->getObjectPtr()) = map_sd;

    if (this->next()->putq(cm2) == -1)
    {
        GERROR("CmrParametricMappingGadget::process, passing sd map on to next gadget");
        return GADGET_FAIL;
    }


    //put data on the gpu

    //perform zerofilling

    // extract partie reelle
    // extract partie imaginaire
    // perform motion correction
    // rassemble les données


    // -------------------------------------------------------------

    /*if (this->next()->putq(m1) == -1)
    {
        GERROR("ZeroFillingGadget::process, passing map on to next gadget");
        return GADGET_FAIL;
    }*/

    m1->release();

    if (perform_timing.value()) { gt_timer_local_.stop(); }

    return GADGET_OK;
}

void ZeroFillingGadget::perform_zerofilling(IsmrmrdImageArray& in, hoNDArray< std::complex<float> > & data_out, int scaling)
{
    try
    {
        if (perform_timing.value()) { gt_timer_.start("ZeroFillingGadget::perform_filling"); }

        GDEBUG_CONDITION_STREAM(verbose.value(), "ZeroFillingGadget::perform_zerofilling(...) starts ... ");

        size_t RO = in.data_.get_size(0);
        size_t E1 = in.data_.get_size(1);
        size_t E2 = in.data_.get_size(2);
        size_t CHA = in.data_.get_size(3);
        size_t N = in.data_.get_size(4);
        size_t S = in.data_.get_size(5);
        size_t SLC = in.data_.get_size(6);

        Gadgetron::hoNDFFT<float>::instance()->fft2c(in.data_);

        unsigned int offset_ro=int(RO/scaling);
        unsigned int offset_e1=int(E1/scaling);
        GDEBUG("offset %d ", offset_ro, offset_e1);

        hoNDArray< std::complex<float> >  data_in(in.data_);

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

        // -------------------------------------------------------------

        if (perform_timing.value()) { gt_timer_.stop(); }
    }
    catch (...)
    {
        GERROR_STREAM("Exceptions happened in ZeroFillingGadget::perform_zerofilling(...) ... ");
    }


}



// ----------------------------------------------------------------------------------------

GADGET_FACTORY_DECLARE(ZeroFillingGadget)

}


#include "GenericCheckSizeDataRefSBGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericCheckSizeDataRefSBGadget::GenericCheckSizeDataRefSBGadget() : BaseClass()
{
}

GenericCheckSizeDataRefSBGadget::~GenericCheckSizeDataRefSBGadget()
{
}

int GenericCheckSizeDataRefSBGadget::process_config(ACE_Message_Block* mb)
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

    size_t NE = h.encoding.size();
    num_encoding_spaces_ = NE;

    return GADGET_OK;
}

int GenericCheckSizeDataRefSBGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericCheckSizeDataRefSBGadget::process"); }

    process_called_times_++;

    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();

    //if (recon_bit_->rbit_.size() > num_encoding_spaces_)
    //{
    //   GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
    //}

    // for every encoding space, prepare the recon_bit_->rbit_[e].ref_

    std::cout << recon_bit_->rbit_.size() << std::endl;

    size_t e;

    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;

        size_t RO;
        size_t E1;
        size_t E2;
        size_t CHA;
        size_t N;
        size_t S;
        size_t SLC;

        if (recon_bit_->rbit_[e].ref_)
        {
           // std::cout << " je suis la structure qui contient les données acs" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].ref_->data_;

            RO = data.get_size(0);
            E1 = data.get_size(1);
            E2 = data.get_size(2);
            CHA = data.get_size(3);
            N = data.get_size(4);
            S = data.get_size(5);
            SLC = data.get_size(6);

            GDEBUG_STREAM("GenericCheckSizeDataRefSBGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
        }

        if (recon_bit_->rbit_[e].sb_)
        {
           // std::cout << " je suis la structure qui contient les données single band" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].sb_->data_;

            RO = data.get_size(0);
            E1 = data.get_size(1);
            E2 = data.get_size(2);
            CHA = data.get_size(3);
            N = data.get_size(4);
            S = data.get_size(5);
            SLC = data.get_size(6);

            GDEBUG_STREAM("GenericCheckSizeDataRefSBGadget - incoming data array sb : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
        }

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
           // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

            RO = data.get_size(0);
            E1 = data.get_size(1);
            E2 = data.get_size(2);
            CHA = data.get_size(3);
            N = data.get_size(4);
            S = data.get_size(5);
            SLC = data.get_size(6);

            GDEBUG_STREAM("GenericCheckSizeDataRefSBGadget - incoming data array data: [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
        }

    }

    if (this->next()->putq(m1) < 0)
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    if (perform_timing.value()) { gt_timer_.stop(); }


    return GADGET_OK;
}



GADGET_FACTORY_DECLARE(GenericCheckSizeDataRefSBGadget)
}

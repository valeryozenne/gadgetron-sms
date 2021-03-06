
#include "GenericReconSizeGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

    GenericReconSizeGadget::GenericReconSizeGadget() : BaseClass()
    {
    }

    GenericReconSizeGadget::~GenericReconSizeGadget()
    {
    }

    int GenericReconSizeGadget::process_config(ACE_Message_Block* mb)
    {
        GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

        return GADGET_OK;
    }

    int GenericReconSizeGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {
        if (perform_timing.value()) { gt_timer_.start("GenericReconSizeGadget::process"); }

        process_called_times_++;

        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
        if (recon_bit_->rbit_.size() > num_encoding_spaces_)
        {
            GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
        }

        // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
        size_t e, n, s, slc;
        for (e = 0; e < recon_bit_->rbit_.size(); e++)
        {
            auto & rbit = recon_bit_->rbit_[e];
            std::stringstream os;
            os << "_encoding_" << e;

            if (recon_bit_->rbit_[e].ref_)
            {
               // std::cout << " je suis la structure qui contient les données acs" << std::endl;

                hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].ref_->data_;
                hoNDArray< ISMRMRD::AcquisitionHeader > headers_ = recon_bit_->rbit_[e].ref_->headers_;
                ISMRMRD::AcquisitionHeader &  curr_headers= headers_(0, 0, 0, 0, 0);

                //std::cout << " repetition " << .idx.repetition << std::endl;

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t N = data.get_size(4);
                size_t S = data.get_size(5);
                size_t SLC = data.get_size(6);

                GDEBUG_STREAM("GenericCheckSizeGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
            }

            if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
            {
               // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

                hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;
                hoNDArray< ISMRMRD::AcquisitionHeader > headers_ = recon_bit_->rbit_[e].data_.headers_;
                ISMRMRD::AcquisitionHeader &  curr_headers = headers_(0, 0, 0, 0, 1);

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t N = data.get_size(4);
                size_t S = data.get_size(5);
                size_t SLC = data.get_size(6);

                size_t hE1=headers_.get_size(0);
                size_t hE2=headers_.get_size(1);
                size_t hN=headers_.get_size(2);
                size_t hS=headers_.get_size(3);
                size_t hSLC=headers_.get_size(4);

                GDEBUG("GenericCheckSizeGadget - |--------------------------------------------------------------------------|\n");
                if (N>1)
                {
                GDEBUG("GenericCheckSizeGadget - detecting single band and multiband data (if splitSMSGadget is OFF), it should be the first repetition\n");
                }

                bool repetition_zero=true;
                bool single_band_data=false;

                size_t ii;
                for (ii=0; ii<recon_bit_->rbit_[e].data_.headers_.get_number_of_elements(); ii++)
                {
                    if( recon_bit_->rbit_[e].data_.headers_(ii).idx.repetition>0 )
                    {
                        GDEBUG_STREAM("GenericCheckSizeGadget - It is not the first repetition, it is the number "<< recon_bit_->rbit_[e].data_.headers_(ii).idx.repetition);
                        repetition_zero=false;
                        break;
                    }
                }

                if (repetition_zero)
                {
                    GDEBUG("GenericCheckSizeGadget - It is the first repetition\n");
                    for (ii=0; ii<recon_bit_->rbit_[e].data_.headers_.get_number_of_elements(); ii++)
                    {
                        if( recon_bit_->rbit_[e].data_.headers_(ii).idx.user[0]==1 )
                        {
                         GDEBUG_STREAM("GenericCheckSizeGadget - Single band data (if splitSMSGadget is ON)");
                         single_band_data=true;
                         break;
                        }
                    }

                    if (single_band_data==false)
                    {
                    GDEBUG_STREAM("GenericCheckSizeGadget - Multiband data (if splitSMSGadget is ON) ");
                    }
                }

                GDEBUG_STREAM("GenericCheckSizeGadget - incoming data array data: [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

                GDEBUG_STREAM("GenericCheckSizeGadget - incoming data array headers: [E1, E2, N, S, LOC] - [" << hE1 << " " << hE2 << " " << hN << " " << hS << " " << hSLC << "]");





            }
        }

        if (perform_timing.value()) { gt_timer_.stop(); }

        if (this->next()->putq(m1) < 0)
        {
            GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
            return GADGET_FAIL;
        }

        return GADGET_OK;
    }

    GADGET_FACTORY_DECLARE(GenericReconSizeGadget)
}

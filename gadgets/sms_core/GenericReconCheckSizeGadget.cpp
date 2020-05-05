
#include "GenericReconCheckSizeGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

    GenericReconCheckSizeGadget::GenericReconCheckSizeGadget() : BaseClass()
    {
    }

    GenericReconCheckSizeGadget::~GenericReconCheckSizeGadget()
    {
    }

    int GenericReconCheckSizeGadget::process_config(ACE_Message_Block* mb)
    {
        GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

        return GADGET_OK;
    }

    int GenericReconCheckSizeGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {
        if (perform_timing.value()) { gt_timer_.start("GenericReconCheckSizeGadget::process"); }

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

                unsigned int rep_max =0;

                for (int ii=0; ii<headers_.get_number_of_elements(); ii++)
                {
                    if( headers_(ii).idx.repetition>0 )
                    {
                        rep_max=headers_(ii).idx.repetition;
                        //GERROR_STREAM("After checking, it is not the first repetition,  "<< );
                        //return GADGET_FAIL;
                    }
                }

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t N = data.get_size(4);
                size_t S = data.get_size(5);
                size_t SLC = data.get_size(6);

                GDEBUG_STREAM("GenericReconCheckSizeGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
            }


            if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
            {
               // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

                hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;
                hoNDArray< ISMRMRD::AcquisitionHeader > headers_ = recon_bit_->rbit_[e].data_.headers_;
                unsigned int rep_max=0;

                for (int ii=0; ii<headers_.get_number_of_elements(); ii++)
                {
                    if( headers_(ii).idx.repetition>0 )
                    {
                        rep_max=headers_(ii).idx.repetition;
                        //GERROR_STREAM("After checking, it is not the first repetition,  "<< );
                        //return GADGET_FAIL;
                    }
                }

                std::cout << " repetition " << rep_max << std::endl;


                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t N = data.get_size(4);
                size_t S = data.get_size(5);
                size_t SLC = data.get_size(6);

                GDEBUG_STREAM("GenericReconCheckSizeGadget - incoming data array data: [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");


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

    GADGET_FACTORY_DECLARE(GenericReconCheckSizeGadget)
}


#include "GenericReconSMSPostGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

    GenericReconSMSPostGadget::GenericReconSMSPostGadget() : BaseClass()
    {
    }

    GenericReconSMSPostGadget::~GenericReconSMSPostGadget()
    {
    }

    int GenericReconSMSPostGadget::process_config(ACE_Message_Block* mb)
    {
        GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

        return GADGET_OK;
    }

    int GenericReconSMSPostGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {
        if (perform_timing.value()) { gt_timer_.start("GenericReconSMSPostGadget::process"); }

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

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t N = data.get_size(4);
                size_t S = data.get_size(5);
                size_t SLC = data.get_size(6);

                GDEBUG_STREAM("GenericReconSMSPostGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
            }

            if (recon_bit_->rbit_[e].sb_)
            {
               // std::cout << " je suis la structure qui contient les données single band" << std::endl;

                hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].sb_->data_;

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t MB = data.get_size(4);
                size_t STK = data.get_size(5);
                size_t N = data.get_size(6);
                size_t S = data.get_size(7);

                GDEBUG_STREAM("GenericReconSMSPostGadget - incoming data array sb : [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N << " " << S <<"]");

                hoNDArray< std::complex<float> > data7D;

                data7D.create(RO, E1, E2, CHA, N, S, STK*MB);

                undo_stacks_ordering_to_match_gt_organisation(data, data7D);

                m1->getObjectPtr()->rbit_[e].sb_->data_ = data7D;

            }

            if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
            {
               // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

                hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t MB = data.get_size(4);
                size_t STK = data.get_size(5);
                size_t N = data.get_size(6);
                size_t S = data.get_size(7);

                GDEBUG_STREAM("GenericReconSMSPostGadget - incoming data array data: [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N << " " << S <<"]");




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



    void GenericReconSMSPostGadget::undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output)
        {
            size_t RO=data.get_size(0);
            size_t E1=data.get_size(1);
            size_t E2=data.get_size(2);
            size_t CHA=data.get_size(3);
            size_t MB=data.get_size(4);
            size_t STK=data.get_size(5);
            size_t N=data.get_size(6);
            size_t S=data.get_size(7);

            //GADGET_CHECK_THROW(lNumberOfSlices_ == STK*MB);

            size_t n, s, a, m;
            int index;

            for (a = 0; a < STK; a++) {

                for (m = 0; m < MB; m++) {

                    index = MapSliceSMS(a,m);

                    for (s = 0; s < S; s++)
                    {

                        for (n = 0; n < N; n++)
                        {                            

                            std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                            std::complex<float> * out = &(output(0, 0, 0, 0, n, s, index));
                            memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                        }
                    }
                }
            }
        }

    GADGET_FACTORY_DECLARE(GenericReconSMSPostGadget)
}

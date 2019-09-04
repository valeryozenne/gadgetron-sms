
#include "GenericCheckSizeSimpleGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericCheckSizeSimpleGadget::GenericCheckSizeSimpleGadget() : BaseClass()
{
}

GenericCheckSizeSimpleGadget::~GenericCheckSizeSimpleGadget()
{
}

int GenericCheckSizeSimpleGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);


    return GADGET_OK;
}



int GenericCheckSizeSimpleGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
       size_t e;

    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

            int repetition = recon_bit_->rbit_[e].data_.headers_[0].idx.repetition;


            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            size_t STACK ;

            if (data.get_number_of_dimensions() == 8)
            {
            STACK = data.get_size(7);
            }

            if (data.get_number_of_dimensions() == 8)
            {
            GDEBUG_STREAM("GenericCheckSizeSimpleGadget - incoming data array : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC<< " " << STACK <<  "]");
            }
            else{
           GDEBUG_STREAM("GenericCheckSizeSimpleGadget - incoming data array : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC<<  "]");
            }

        }
    }



    if (this->next()->putq(m1) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(GenericCheckSizeSimpleGadget)
}


#include "GenericChangeSizeTo7Gadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericChangeSizeTo7Gadget::GenericChangeSizeTo7Gadget() : BaseClass()
{
}

GenericChangeSizeTo7Gadget::~GenericChangeSizeTo7Gadget()
{
}

int GenericChangeSizeTo7Gadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}



int GenericChangeSizeTo7Gadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
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

            int dim = data.get_number_of_dimensions();

            if (dim == 8) {

                int nb_of_elements = data.get_number_of_elements();

                hoNDArray< std::complex<float> > new_data;

                new_data.create(data.get_size(0), data.get_size(1), data.get_size(2), data.get_size(3), data.get_size(4), data.get_size(5), data.get_size(6)*data.get_size(7));

                int i;

                for (i = 0; i < nb_of_elements; i++) {

                    new_data[i] = data[i];

                }

                m1->getObjectPtr()->rbit_[e].data_.data_ = new_data;
            }



            int repetition = recon_bit_->rbit_[e].data_.headers_[0].idx.repetition;



        }
    }



    if (this->next()->putq(m1) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(GenericChangeSizeTo7Gadget)
}


#include "GenericCheckSizeGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericCheckSizeGadget::GenericCheckSizeGadget() : BaseClass()
{
}

GenericCheckSizeGadget::~GenericCheckSizeGadget()
{
}

int GenericCheckSizeGadget::process_config(ACE_Message_Block* mb)
{

    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}



int GenericCheckSizeGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{

    IsmrmrdReconData* recon_bit_1 = m1->getObjectPtr();

    size_t e;




    for (e = 0; e < recon_bit_1->rbit_.size(); e++)
    {
        auto & rbit1 = recon_bit_1->rbit_[e];

    hoNDArray< std::complex<float> >& data1 = rbit1.data_.data_;

    int nb_dimensions_1 = data1.get_number_of_dimensions();

    GDEBUG_STREAM(" Check Size incoming data array m1 of " << data1.get_number_of_dimensions() << " dimensions");

    for (int i = 0; i < nb_dimensions_1; i++) {

        GDEBUG_STREAM(" Check Size incoming data array m1 :  [" << data1.get_size(i) << "]");
    }

    if (this->next()->putq(m1) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
    }
}




int GenericCheckSizeGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1, Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m2)
{
    IsmrmrdReconData* recon_bit_1 = m1->getObjectPtr();
    IsmrmrdReconData* recon_bit_2 = m2->getObjectPtr();

    size_t e;




    for (e = 0; e < recon_bit_1->rbit_.size(); e++)
    {
        auto & rbit1 = recon_bit_1->rbit_[e];
        auto & rbit2 = recon_bit_2->rbit_[e];


    hoNDArray< std::complex<float> >& data1 = rbit1.data_.data_;

    hoNDArray< std::complex<float> >& data2 = rbit2.data_.data_;

    int nb_dimensions_1 = data1.get_number_of_dimensions();
    int nb_dimensions_2 = data2.get_number_of_dimensions();


    GDEBUG_STREAM(" incoming data array m1 of " << data1.get_number_of_dimensions() << " dimensions");
    GDEBUG_STREAM(" incoming data array m2 of " << data2.get_number_of_dimensions() << " dimensions");


    for (int i = 0; i < nb_dimensions_1; i++) {

        GDEBUG_STREAM(" incoming data array m1 :  [" << data1.get_size(i) << "]");
    }
    for (int i = 0; i < nb_dimensions_2; i++) {

        GDEBUG_STREAM(" incoming data array m2 :  [" << data2.get_size(i) << "]");
    }

    }

    if (this->next()->putq(m1) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(GenericCheckSizeGadget)
}

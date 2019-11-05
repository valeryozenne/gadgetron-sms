
#include "GenericConcateneGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericConcateneGadget::GenericConcateneGadget() : BaseClass()
{
}

GenericConcateneGadget::~GenericConcateneGadget()
{
}

int GenericConcateneGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);


    //buffer.create(R0,E1,E2)

    compteur=0;


    return GADGET_OK;
}



int GenericConcateneGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();

    std::cout << recon_bit_->rbit_.size() << std::endl;

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
            //int scan_counter= recon_bit_->rbit_[e].data_.headers_[0].scan_counter;
            //int center_sample= recon_bit_->rbit_[e].data_.headers_[0].center_sample;

            compteur++;

            std::cout << "GenericConcateneGadget  repetition" << repetition  << std::endl;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

           GDEBUG_STREAM("GenericConcateneGadget - incoming data array : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC <<  "]");

            if (compteur==1)
            {
                std::cout << " compteur==1  1 er paquet " << std::endl;



                buffer.create(RO, E1, E2, CHA, N,S, SLC);

               // & buffer = recon_bit_->rbit_[e].data_.data_;

            }
            else if (compteur==2)
            {
                std::cout << " compteur==2 2 nd  paquet " << std::endl;
                compteur=0;


                hoNDArray< std::complex<float> >& buffer2 = recon_bit_->rbit_[e].data_.data_;

                //concatenatation de buffer et buffer 2


                // hoNDArray< std::complex<float> > data ;
                // data.createRO, E1, E2, CHA1+CHA2, N,S, SLC);



                // on renvoit les données au gadget suivant
                // on cree un gadget messager m2
                // on associe data au gadget messager

                /* if (this->next()->putq(m2) < 0 )
                 {
                     GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                     return GADGET_FAIL;
                 }*/

            }
            else
            {
                std::cout << " pb  " << std::endl;
            }

        }
    }


    //m1->release;

   /* if (this->next()->putq(m1) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }*/

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(GenericConcateneGadget)
}

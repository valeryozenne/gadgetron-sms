
#include "GenericEmptyTestGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericEmptyTestGadget::GenericEmptyTestGadget() : BaseClass()
{
}

GenericEmptyTestGadget::~GenericEmptyTestGadget()
{
}

int GenericEmptyTestGadget::process_config(ACE_Message_Block* mb)
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



int GenericEmptyTestGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{


    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();

    //doesn't work, to send 2 messages we have to inherit from Gadget2 and not Gadget1

   Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m_1 = m1->duplicate();


    Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m_2 = m1->duplicate();


    size_t e;




    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;


        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {


            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;


            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3)-1;
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            GDEBUG_STREAM("GenericEmptyTestGadget - incoming data array : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC <<  "]");


            //creation vecteur multidimensionnel
            hoNDArray< std::complex<float> > dataCha;
            hoNDArray< std::complex<float> > dataCha2;

            //gerer le cas antennes impair
            // Séparer en 2 hoNDArray selon la taille de CHA
            if (CHA%2 != 0) {
              dataCha2.create(RO, E1, E2, CHA/2 + 1, N, S, SLC);
            }

            else {

              dataCha2.create(RO, E1, E2, CHA/2, N, S, SLC);

            }
            dataCha.create(RO, E1, E2, CHA/2 , N, S, SLC);



            data.get_dimensions();
            int CHA_size, CHA2_size, size_before_CHA, size_after_CHA, i;
            CHA_size = dataCha.get_size(3);
            CHA2_size = dataCha2.get_size(3);
            size_before_CHA = data.get_size(0)*data.get_size(1)*data.get_size(2);
            size_after_CHA = data.get_size(4) * data.get_size(5) * data.get_size(6);

            //bad copy, don't do this :
            for (i = 0; i < size_before_CHA; i++) {
                dataCha[i] = data[i];
                dataCha2[i] = data[i];
            }

            for (i = 0; i < size_after_CHA; i++) {
                dataCha[size_before_CHA + dataCha.get_size(3) + i] = data[size_before_CHA + data.get_size(3) + i];
                dataCha2[size_before_CHA + dataCha2.get_size(3)+ i] = data[size_before_CHA + data.get_size(3) + i];
            }


            for (i = 0; i < CHA_size; i++) {
                dataCha[size_before_CHA + i] = data[size_before_CHA + i];
            }

            for (i = 0; i < CHA2_size; i++) {
                dataCha2[size_before_CHA + i] = data[size_before_CHA + dataCha.get_size(3) + i];
            }


            // affectation of the new data  the new message





            m_1->getObjectPtr()->rbit_[e].data_.data_ = dataCha2;
            m_2->getObjectPtr()->rbit_[e].data_.data_ = dataCha;



            //problème : affectation sur la même adresse/pointeur


            std::cout << "adresse data message 1 : " << &m_1->getObjectPtr()->rbit_[e].data_.data_ << std::endl;
            std::cout << "adresse data message 2 : " <<  &m_2->getObjectPtr()->rbit_[e].data_ << std::endl;


        }

    }




    if (this->next()->putq(m_1) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }



    if (this->next()->putq(m_2) < 0 )
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(GenericEmptyTestGadget)
}

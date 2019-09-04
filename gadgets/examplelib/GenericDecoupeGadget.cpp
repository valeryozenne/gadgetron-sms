
#include "GenericDecoupeGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericDecoupeGadget::GenericDecoupeGadget() : BaseClass()
{
}

GenericDecoupeGadget::~GenericDecoupeGadget()
{
}

int GenericDecoupeGadget::process_config(ACE_Message_Block* mb)
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



int GenericDecoupeGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
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
           // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

             size_t RO = data.get_size(0);
             size_t E1 = data.get_size(1);
             size_t E2 = data.get_size(2);
             size_t CHA = data.get_size(3);
             size_t N = data.get_size(4);
             size_t S = data.get_size(5);
             size_t SLC = data.get_size(6);


             size_t CHA1, CHA2;

             if (CHA%2 != 0) {

               CHA1=CHA/2;
               CHA2=CHA/2+1;

             }

             else {


                 CHA1=CHA/2;
                 CHA2=CHA/2;


             }


            bool cut_channel=1;

            if (cut_channel)
            {
                //on va devoir revoyer deux flux
                // m1 avec les données de m1 à 0 a CHA/2+1
                // m2 avec les données de m1 de CHA/2+1 a CHA


                //creation vecteur multidimensionnel
                hoNDArray< std::complex<float> > data1;
                hoNDArray< std::complex<float> > data2;

                data1.create(RO, E1, E2, CHA1, N, S, SLC);
                data2.create(RO, E1, E2, CHA2, N, S, SLC);

                size_t RO_t = data1.get_size(0);
                size_t E1_t = data1.get_size(1);
                size_t E2_t = data1.get_size(2);
                size_t CHA_t = data1.get_size(3);
                size_t N_t = data1.get_size(4);
                size_t S_t = data1.get_size(5);
                size_t SLC_t = data1.get_size(6);

                 std::cout << " ok 1" << std::endl;
                //  copie des données

                 GDEBUG_STREAM("GenericEmptyTestGadget - incoming data array : [RO E1 E2 CHA N S SLC] - [" << RO_t << " " << E1_t << " " << E2_t << " " << CHA_t << " " << N_t << " " << S_t << " " << SLC_t <<  "]");


                // on renvoie m1 au gadget suivant


                // je duplique m1 en m2
                //Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m2 = m1->duplicate();
                Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m2 = new Gadgetron::GadgetContainerMessage< IsmrmrdReconData >();



                /*m2->getObjectPtr()->rbit_;

                IsmrmrdReconBit *rbit2 = new std::vector<IsmrmrdReconBit>();

                IsmrmrdDataBuffered *data2buff = new IsmrmrdDataBuffered();
                hoNDArray< std::complex<float> > *data2array = new hoNDArray< std::complex<float> >();

                m2->getObjectPtr()->rbit_ = *rbit2;
                rbit2.data_ = data2buff;
                rbit2[e].data_.data_ = data2array;

                m2->getObjectPtr()->rbit_[e].data_.data_ = &data2array;*/

                //IsmrmrdReconBit test;
                //IsmrmrdReconData test;



                 m1->getObjectPtr()->rbit_[e].data_.data_ = data1;
                 m2->getObjectPtr()->rbit_[e].data_.data_ = data2;

                 //m_1->getObjectPtr()->rbit_[e].data_.headers_=


                // on renvoie m2 au gadget suivant
              /*  if (this->next()->putq(m2) < 0 )
                {
                    GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                    return GADGET_FAIL;
                }*/


                if (this->next()->putq(m1) < 0 )
                {
                    GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                    return GADGET_FAIL;
                }


            }
            else
            {

                // on renvoit m1 tel que et on fait rien d'autre

                if (this->next()->putq(m1) < 0 )
                 {
                     GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                     return GADGET_FAIL;
                 }

            }






        }
    }





    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(GenericDecoupeGadget)
}

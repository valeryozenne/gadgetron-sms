#include "EmptyRemoveROOversamplingGadget.h"
#include "hoNDFFT.h"
#include "ismrmrd/xml.h"

#ifdef USE_OMP
    #include "omp.h"
#endif // USE_OMP

namespace Gadgetron{

    EmptyRemoveROOversamplingGadget::EmptyRemoveROOversamplingGadget()
    {
    }

    EmptyRemoveROOversamplingGadget::~EmptyRemoveROOversamplingGadget()
    {
    }

    int EmptyRemoveROOversamplingGadget::process_config(ACE_Message_Block* mb)
    {

	ISMRMRD::IsmrmrdHeader h;
	ISMRMRD::deserialize(mb->rd_ptr(),h);



        return GADGET_OK;
    }

    int EmptyRemoveROOversamplingGadget
        ::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
        GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
    {

        int scan_counter= m1->getObjectPtr()->scan_counter;
        int center_sample= m1->getObjectPtr()->center_sample;
        int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
        int e2= m1->getObjectPtr()->idx.kspace_encode_step_2;
        int slice= m1->getObjectPtr()->idx.slice;
        int average= m1->getObjectPtr()->idx.average;
        int repetition= m1->getObjectPtr()->idx.repetition;

      //  std::cout << " scan_counter "<< scan_counter<< "  e1 "<< e1 <<  "  e2 "<< e2  <<  "  repetition "<< repetition<<  "  average "<< average << std::endl;


        //m2 a deux dimensions [R0 , CHA]
         int RO=m2->getObjectPtr()->get_size(0);
          int CHA=m2->getObjectPtr()->get_size(1);


       //   std::cout << " RO "<< RO<< "CHA "<< CHA << std::endl;


      if (this->next()->putq(m1) == -1)
      {
    GERROR("EmptyRemoveROOversamplingGadget::process, passing data on to next gadget");
	return GADGET_FAIL;
      }

      return GADGET_OK;
    }


    GADGET_FACTORY_DECLARE(EmptyRemoveROOversamplingGadget)
}


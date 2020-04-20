#include "GiveReadoutInformationGadget.h"

namespace Gadgetron{



GiveReadoutInformationGadget::GiveReadoutInformationGadget()
{
}

GiveReadoutInformationGadget::~GiveReadoutInformationGadget()
{
}

int GiveReadoutInformationGadget::process_config(ACE_Message_Block* mb)
{

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



    return GADGET_OK;
}



int GiveReadoutInformationGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
                                          GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{
    //It is enough to put the first one, since they are linked


    bool is_parallel_calibration =  m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);
    bool is_phase_corr =  m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA);


    if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION))
    {

        if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
        {
        //do something  // 3 lignes
        }
        else
        {
        //do something  //24 lignes
        }
    }
    else
    {
        if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
        {
        //do something  // 3 lignes
        }
        else
        {
        //do something si grappa // une ligne sur deux  // 128/2
        }
    }

    display_header_information(m1 , is_parallel_calibration , is_phase_corr);


    if (this->next()->putq(m1) == -1) {
        m1->release();
        GERROR("AcquisitionPassthroughGadget::process, passing data on to next gadget");
        return -1;
    }

    return 0;
}


void GiveReadoutInformationGadget::display_header_information(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1, bool is_calib, bool is_phase_corr)
{

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int e2= m1->getObjectPtr()->idx.kspace_encode_step_2;
    unsigned int slice= m1->getObjectPtr()->idx.slice;
    unsigned int repetition= m1->getObjectPtr()->idx.repetition;
    unsigned int set= m1->getObjectPtr()->idx.set;
    unsigned int segment= m1->getObjectPtr()->idx.segment;
    unsigned int phase= m1->getObjectPtr()->idx.phase;
    unsigned int average= m1->getObjectPtr()->idx.average;
    unsigned int user= m1->getObjectPtr()->idx.user[0];
    unsigned int scan_counter=m1->getObjectPtr()->scan_counter;

    std::string message1;
    std::string message2;


    if (is_calib)
    {

        message1= "CALIB";

        if (is_phase_corr)
        {
           message2="PHAS_CORR";
        }
        else
        {
           message2="DATA";
        }
    }
    else
    {
        message1= "IMAGING";

        if (is_phase_corr)
        {
                message2="PHAS_CORR";
        }
        else
        {
               message2="DATA";
        }
    }

    std::cout << "--------------------------------------"<< std::endl;
    std::cout << " This a new readout, n° " << scan_counter <<" The data is " <<message1<< " + "   << message2 << std::endl;
    std::cout << " Encoding direction are e1: " << e1 << " e2: "<< e2 <<" slc: "<< slice << " rep: "<< repetition << " seg: "<< segment << " user: "<<user<< std::endl;

}


GADGET_FACTORY_DECLARE(GiveReadoutInformationGadget)
}



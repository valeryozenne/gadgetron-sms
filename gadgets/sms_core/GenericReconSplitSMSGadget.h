/** \file   GenericReconSplitSMSGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

#include "GenericReconBase.h"
//#include "GenericReconGadget.h"
#include "gadgetron_smscore_export.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"

namespace Gadgetron {

class EXPORTGADGETSSMSCORE GenericReconSplitSMSGadget : public GenericReconDataBase
{
public:
    GADGET_DECLARE(GenericReconSplitSMSGadget);

    typedef GenericReconDataBase BaseClass;

    GenericReconSplitSMSGadget();
    ~GenericReconSplitSMSGadget();

    GADGET_PROPERTY(use_omp, bool, "Whether to use omp acceleration", false);

    /// ------------------------------------------------------------------------------------
    /// parameters to control the reconstruction
    /// ------------------------------------------------------------------------------------


protected:


    // --------------------------------------------------
    // gadget functions
    // --------------------------------------------------
    // default interface function
    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
    virtual void extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb);
    virtual void extract_sb_and_mb_from_data_open(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb);

    hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_mb;

    hoNDArray< std::complex<float> > sb;
    hoNDArray< std::complex<float> > mb;

};
}

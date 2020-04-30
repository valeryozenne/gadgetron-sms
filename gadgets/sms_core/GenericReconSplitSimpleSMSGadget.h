/** \file   GenericReconSplitSimpleSMSGadget.h
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

class EXPORTGADGETSSMSCORE GenericReconSplitSimpleSMSGadget : public GenericReconDataBase
{
public:
    GADGET_DECLARE(GenericReconSplitSimpleSMSGadget);

    typedef GenericReconDataBase BaseClass;

    GenericReconSplitSimpleSMSGadget();
    ~GenericReconSplitSimpleSMSGadget();

    GADGET_PROPERTY(use_omp, bool, "Whether to use omp acceleration", false);
    //GADGET_PROPERTY(debug_folder, std::string, "To use std::copy or memcpy", "/tmp/gadgetron");

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
    virtual void extract_sb_and_mb_from_data_memcpy(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb);
    void extract_sb_and_mb_from_data_std_cpy(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb);
    virtual void extract_sb_and_mb_from_data_open(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb);
    void compareData(hoNDArray< std::complex<float> > &data, hoNDArray< std::complex<float> > &sb, hoNDArray< std::complex<float> > &mb);
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_mb;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_mb_std_copy;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb_std_copy;

    hoNDArray< std::complex<float> > sb;
    hoNDArray< std::complex<float> > mb;
    hoNDArray< std::complex<float> > sb_std_copy;
    hoNDArray< std::complex<float> > mb_std_copy;

    bool is_first_repetition;

};
}

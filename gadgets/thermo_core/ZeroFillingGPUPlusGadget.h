/**
\file   ZeroFillingGPUPlusGadget.h
\brief  This is the class gadget for cardiac T2 mapping, working on the IsmrmrdImageArray.
\author Hui Xue
*/

#pragma once

#include "GenericReconBase.h"
#include "hoNDFFT.h"
#include "hoNDArray_math.h"
#include "gadgetron_thermocore_export.h"
#include "cuNDArray.h"
#include "cuNDFFT.h"
namespace Gadgetron {

    class EXPORTGADGETSTHERMOCORE ZeroFillingGPUPlusGadget : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(ZeroFillingGPUPlusGadget);

        typedef GenericReconImageBase BaseClass;

        ZeroFillingGPUPlusGadget();
        ~ZeroFillingGPUPlusGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the mapping
        /// ------------------------------------------------------------------------------------
        GADGET_PROPERTY(oversampling, int, "Whether to send out gfactor map", 1);
        GADGET_PROPERTY(use_gpu, bool, "Whether to use gpu forr ifft", false);


    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------
        // hoNDArray< complext<float> > data_in;
        // hoNDArray< complext<float> > data_out;

        cuNDArray< complext<float> > cu_data;
        cuNDArray< complext<float> > cu_data_out;
        // --------------------------------------------------
        // functional functions
        // --------------------------------------------------

        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);

        // function to perform the mapping
        // data: input image array [RO E1 E2 CHA N S SLC]
        // map and map_sd: mapping result and its sd
        // para and para_sd: other parameters of mapping and its sd
        //virtual int perform_zerofilling(IsmrmrdImageArray& data, IsmrmrdImageArray& map);
        virtual void perform_zerofilling(hoNDArray< std::complex<float> > & data_in, hoNDArray< std::complex<float> > & data_out);
        virtual void perform_zerofilling_array(IsmrmrdImageArray& in, IsmrmrdImageArray& out);
        virtual void perform_zerofilling_array_gpu(IsmrmrdImageArray& in, IsmrmrdImageArray& out);
         //virtual void perform_zerofilling(IsmrmrdImageArray& in, IsmrmrdImageArray& out);
        void perform_zerofilling_gpu(cuNDArray<complext<float>> in, cuNDArray<complext<float>> out);


  };
}

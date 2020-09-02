/**
\file   SortSliceGadget.h
\brief  This is the class gadget for cardiac T2 mapping, working on the IsmrmrdImageArray.
\author Hui Xue
*/

#pragma once


#include "GenericReconBase.h"
#include "hoNDFFT.h"
#include "hoNDArray_math.h"
#include "gadgetron_smscore_export.h"
//#include "mri_core_slice_grappa.h"
#include "hoArmadillo.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE SortSliceGadget : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(SortSliceGadget);

        typedef GenericReconImageBase BaseClass;

        SortSliceGadget();
        ~SortSliceGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the mapping
        /// ------------------------------------------------------------------------------------
        GADGET_PROPERTY(oversampling, unsigned int, "Whether to send out gfactor map", 1);


    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // --------------------------------------------------
        // functional functions
        // --------------------------------------------------

        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);

        virtual std::vector<unsigned int> map_interleaved_acquisitions(int number_of_slices, bool no_reordering );
        virtual std::vector<unsigned int> sort_index(std::vector<unsigned int>data);



        size_t counter_;

       std::vector<size_t> dimensions_;


        bool is_wip_sequence ;
        bool is_cmrr_sequence ;

        unsigned int MB_factor;
        int Blipped_CAIPI;

        unsigned int lNumberOfStacks_;
        unsigned int lNumberOfSlices_;
        unsigned int lNumberOfChannels_;

        // acceleration factor for E1 and E2
        std::vector<double> acceFactorSMSE1_;
        std::vector<double> acceFactorSMSE2_;

        // ordre des coupes
        std::vector< std::vector<unsigned int> > MapSliceSMS;
        std::vector<unsigned int> order_of_acquisition_sb;
        std::vector<unsigned int> order_of_acquisition_mb;

        std::vector<unsigned int> indice_mb;
        std::vector<unsigned int> indice_sb;
        std::vector<unsigned int> indice_slice_mb;

        // function to perform the mapping
        // data: input image array [RO E1 E2 CHA N S SLC]
        // map and map_sd: mapping result and its sd
        // para and para_sd: other parameters of mapping and its sd
        //virtual int perform_zerofilling(IsmrmrdImageArray& data, IsmrmrdImageArray& map);
        //virtual void perform_zerofilling(IsmrmrdImageArray& in, hoNDArray< std::complex<float> > & data_out);
        //virtual void perform_zerofilling(IsmrmrdImageArray& in, IsmrmrdImageArray& out);


  };
}

/** \file   GenericMBReconDataBase.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to prepare the reference data, working on the IsmrmrdReconData.
    \author Hui Xue
*/

#pragma once

#include "gadgetron_examplelib_export.h"
#include "GenericReconBase.h"


#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"

namespace Gadgetron {

    class EXPORTGADGETSEXAMPLELIB GenericMBReconDataBase : public GenericReconDataBase
    {
    public:
        GADGET_DECLARE(GenericMBReconDataBase);

        typedef GenericReconDataBase BaseClass;

        GenericMBReconDataBase();
        ~GenericMBReconDataBase();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------

        /// ref preparation
        /// whether to average all N for ref generation
        /// for the interleaved mode, the sampling times will be counted and used for averaging
        /// it is recommended to set N as the interleaved dimension
        GADGET_PROPERTY(average_all_ref_N, bool, "Whether to average all N for ref generation", true);
        /// whether to average all S for ref generation
        GADGET_PROPERTY(average_all_ref_S, bool, "Whether to average all S for ref generation", false);
        /// whether to update ref for every incoming IsmrmrdReconData; for some applications, we may want to only compute ref data once
        /// if false, the ref will only be prepared for the first incoming IsmrmrdReconData
        GADGET_PROPERTY(prepare_ref_always, bool, "Whether to prepare ref for every incoming IsmrmrdReconData", true);

    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        unsigned int MB_factor_;
        unsigned int number_of_slices;
        unsigned int number_of_stacks;
        unsigned int number_of_channels;
        std::vector<int> order_of_acquisition_sb;
        std::vector<int> order_of_acquisition_mb;
        std::vector<int> indice_sb;
        std::vector<int> indice_mb;
        std::vector<int> indice_slice_mb;
        std::vector< std::vector<int>> MapSliceSMS;
        std::vector<int> vec_MapSliceSMS;

        /// indicate whether ref has been prepared for an encoding space
        std::vector<bool> ref_prepared_;

        // for every encoding space
        // calibration mode
        std::vector<Gadgetron::ismrmrdCALIBMODE> calib_mode_;

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

        //prep fonctions

        /*remove extra dimension of data : (a, b, c, d, e, f, 27) -> (a, b, c, d, e, f, 18)
        imput : hoNDarray of 7 dimensions where the size of the 7th dimension is 27
        output : hoNDarray of 7 dimensions where the size of the 7th dimension is 18

        The function remove extra dimension in the multiband buffer as the
        kspace data is store by the scanner using non consecutive slice
        as an example with MB 2, 12 slices using the WIP from Siemens:
        MB data are store in slice 1 2 3 and 7 8 9,
        the current implementation of the gadgetron therefore create a container
        with slice dimension of 9 that as to be reduced to 6
        finally: the fonction reorganize the MB slice in order to match the
        simulated MB data and calculate correctly the kernel*/

        //input : [RO E1 E2 CHA N S SLC]
        //output : [RO E1 E2 CHA N S STACK SLICES]
        //create a new data array with stacks and slices dimensions where the multiband factor is equal to the size of SLICES's dimension and size(STACK) * size(SLICES) = size(SLC);
        //virtual hoNDArray< std::complex<float> > create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& fid_stack_SB);

    };

}


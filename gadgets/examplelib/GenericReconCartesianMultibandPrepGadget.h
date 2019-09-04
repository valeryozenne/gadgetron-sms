/** \file   GenericReconCartesianMultibandPrepGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to prepare the reference data, working on the IsmrmrdReconData.
    \author Hui Xue
*/

#pragma once

#include "GenericMBReconDataBase.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"

namespace Gadgetron {

    class EXPORTGADGETSEXAMPLELIB GenericReconCartesianMultibandPrepGadget : public GenericMBReconDataBase
    {
    public:
        GADGET_DECLARE(GenericReconCartesianMultibandPrepGadget);

        typedef GenericMBReconDataBase BaseClass;

        GenericReconCartesianMultibandPrepGadget();
        ~GenericReconCartesianMultibandPrepGadget();

    protected:


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

        virtual void remove_extra_dimension(hoNDArray< std::complex<float> >& data);

        //reorganize the SLC dimension (the 7th) and the size of the 7th has to be equal to 27

        virtual void reorganize_data(hoNDArray< std::complex<float> >& data, std::vector<int> indice);

        //input : [RO E1 E2 CHA N S SLC]
        //output : [RO E1 E2 CHA N S STACK SLICES]
        //create a new data array with stacks and slices dimensions where the multiband factor is equal to the size of SLICES's dimension and size(STACK) * size(SLICES) = size(SLC);

        virtual void create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& fid_stack_SB);

        virtual void export_data_for_each_slice(hoNDArray< std::complex<float> >& data , std::string filename,  size_t e);

        virtual void export_data_for_each_stack_and_slice(hoNDArray< std::complex<float> >& data , std::string filename,  size_t e);

    };
}


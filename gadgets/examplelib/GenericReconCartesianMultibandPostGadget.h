/** \file   GenericReconCartesianMultibandPostGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to prepare the reference data, working on the IsmrmrdReconData.
    \author Hui Xue
*/

#pragma once
#include "GenericMBReconDataBase.h"

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"

namespace Gadgetron {

    class EXPORTGADGETSEXAMPLELIB GenericReconCartesianMultibandPostGadget : public GenericMBReconDataBase
    {
    public:
        GADGET_DECLARE(GenericReconCartesianMultibandPostGadget);

        typedef GenericMBReconDataBase BaseClass;

        GenericReconCartesianMultibandPostGadget();
        ~GenericReconCartesianMultibandPostGadget();

    protected:

        // gadget functions
        // --------------------------------------------------
        virtual void undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& output);



        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);




    };
}

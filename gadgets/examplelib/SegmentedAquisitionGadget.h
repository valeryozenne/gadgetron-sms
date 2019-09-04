//SegmentedAquisitionGadget.h

#ifndef SegmentedAquisitionGadget_H
#define SegmentedAquisitionGadget_H

#include "gadgetron_examplelib_export.h"

#include "GenericReconBase.h"
#include "mri_core_def.h"
//#include "GadgetIsmrmrdReadWrite.h"
#include "Gadget.h"

#include "hoNDArray.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDFFT.h"

#define USE_ARMADILLO 1
#include "hoArmadillo.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_fileio.h"
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include <complex>
#include <map>
#include <armadillo>


#define _USE_MATH_DEFINES
#include <math.h>

namespace Gadgetron
{

  class EXPORTGADGETSEXAMPLELIB SegmentedAquisitionGadget : public GenericReconDataBase
    {
    public:
      GADGET_DECLARE(SegmentedAquisitionGadget);

      typedef GenericReconDataBase BaseClass;

      SegmentedAquisitionGadget();
      virtual ~SegmentedAquisitionGadget();
      
      GADGET_PROPERTY_LIMITS(Sampling_Scheme, std::string, "Sampling_Scheme", "Basic",
            GadgetPropertyLimitsEnumeration, "Basic", "RIGID");
    protected:
      hoNDArray< std::complex<float> > ref_shot_buf_data;
      hoNDArray< std::complex<float> > ref_shot_buf_head;
      
      bool ref_data ;

      virtual int process_config(ACE_Message_Block* mb);
      virtual int process(GadgetContainerMessage<IsmrmrdReconData>* m1);
    };
}
#endif //SegmentedAquisitionGadget_H

#include "ismrmrd.h"
#include "hoNDArray.h"

#pragma once

#include "EPIExport.h"

namespace Gadgetron {
	typedef struct s_EPICorrection
	{
		ISMRMRD::AcquisitionHeader hdr;
		Gadgetron::hoNDArray< std::complex<float> > correction;
	}t_EPICorrection;
}

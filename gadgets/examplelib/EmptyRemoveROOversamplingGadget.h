#pragma once

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_examplelib_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

    class EXPORTGADGETSEXAMPLELIB EmptyRemoveROOversamplingGadget :
        public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
    {
    public:
        GADGET_DECLARE(EmptyRemoveROOversamplingGadget);

        EmptyRemoveROOversamplingGadget();
        virtual ~EmptyRemoveROOversamplingGadget();

    protected:

        virtual int process_config(ACE_Message_Block* mb);

        virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
            GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);

        /*hoNDArray< std::complex<float> > fft_res_;
        hoNDArray< std::complex<float> > ifft_res_;

        hoNDArray< std::complex<float> > fft_buf_;
        hoNDArray< std::complex<float> > ifft_buf_;

        int   encodeNx_;
        float encodeFOV_;
        int   reconNx_;
        float reconFOV_;

	// if true the gadget performs the operation
	// otherwise, it just passes the data on
	bool dowork_;
    */
    };

}

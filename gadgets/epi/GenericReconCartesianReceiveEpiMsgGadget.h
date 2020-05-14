#ifndef GENERICRECONARTESIANRECEIVEEPIMSGGADGET_H
#define GENERICRECONARTESIANRECEIVEEPIMSGGADGET_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "hoArmadillo.h"
#include "gadgetron_epi_export.h"
#include "SMS_utils.h"
#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"
#include <complex>
#include "mri_core_data.h"

#define _USE_MATH_DEFINES

#include <math.h>

namespace Gadgetron {

class EXPORTGADGETS_EPI GenericReconCartesianReceiveEpiMsgGadget :
        public Gadget1<IsmrmrdReconData> {
public:
    GenericReconCartesianReceiveEpiMsgGadget();

    virtual ~GenericReconCartesianReceiveEpiMsgGadget();

protected:
    GADGET_PROPERTY(verboseMode, bool, "Verbose output", false);
    GADGET_PROPERTY(referenceNavigatorNumber, size_t,
                    "Navigator number to be used as reference, both for phase correction and weights for filtering (default=1 -- second navigator)",
                    1);
    GADGET_PROPERTY_LIMITS(B0CorrectionMode, std::string, "B0 correction mode", "mean",
                           GadgetPropertyLimitsEnumeration,
                           "none",
                           "mean",
                           "linear");
    GADGET_PROPERTY_LIMITS(OEPhaseCorrectionMode, std::string, "Odd-Even phase-correction mode", "polynomial",
                           GadgetPropertyLimitsEnumeration,
                           "none",
                           "mean",
                           "linear",
                           "polynomial");
    GADGET_PROPERTY(navigatorParameterFilterLength, int,
                    "Number of repetitions to use to filter the navigator parameters (set to 0 or negative for no filtering)",
                    0);
    GADGET_PROPERTY(navigatorParameterFilterExcludeVols, size_t,
                    "Number of volumes/repetitions to exclude from the beginning of the run when filtering the navigator parameters (e.g., to take into account dummy acquisitions. Default: 0)",
                    0);

    virtual int process_config(ACE_Message_Block *mb);

    virtual int process(
            GadgetContainerMessage<IsmrmrdReconData> *m2);

    /*virtual int process(
            GadgetContainerMessage<t_EPICorrection > *m2);*/

    // in verbose mode, more info is printed out
    bool verboseMode_;

    // --------------------------------------------------
    // variables for navigator parameter computation
    // --------------------------------------------------

    float RefNav_to_Echo0_time_ES_; // Time (in echo-spacing uints) between the reference navigator and the f\
    irst RO echo (used for B0 correction)
    arma::cx_fvec corrB0_;      // B0 correction
    arma::cx_fvec corrpos_;     // Odd-Even correction -- positive readouts
    arma::cx_fvec corrneg_;     // Odd-Even correction -- negative readouts
    arma::cx_fcube navdata_;

    // epi parameters
    int numNavigators_;
    int etl_;

    // for a given shot
    bool corrComputed_;
    int navNumber_;
    int epiEchoNumber_;
    bool startNegative_;


    std::vector<std::pair<GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *, GadgetContainerMessage<hoNDArray<std::complex<float> > > *>> unprocessed_data;

    // --------------------------------------------------
    // variables for navigator parameter filtering
    // --------------------------------------------------

    arma::fvec t_;        // vector with repetition numbers, for navigator filtering
    size_t E2_;       // number of kspace_encoding_step_2
    std::vector<std::vector<size_t> > excitNo_;  // Excitation number (for each set and slice)

    // arrays for navigator parameter filtering:

    hoNDArray<float> Nav_mag_;      // array to store the average navigator magnitude
    //hoNDArray<float>    Nav_phi_;      // array to store the average navigator phase (for 3D or multi-shot imaging)
    hoNDArray<float> B0_slope_;     // array to store the B0-correction linear   term (for filtering)
    hoNDArray<float> B0_intercept_; // array to store the B0-correction constant term (for filtering)
    hoNDArray<float> OE_phi_slope_;     // array to store the Odd-Even phase-correction linear   term (for filtering)
    hoNDArray<float> OE_phi_intercept_; // array to store the Odd-Even phase-correction constant term (for filtering)
    std::vector<hoNDArray<float> > OE_phi_poly_coef_;   // vector of arrays to store the polynomial coefficients for Odd-Even phase correction

};
}
#endif //GENERICRECONARTESIANRECEIVEEPIMSGGADGET_H

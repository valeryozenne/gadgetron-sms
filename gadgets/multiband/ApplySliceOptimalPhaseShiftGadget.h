#ifndef ApplySliceOptimalPhaseShiftGadget_H_
#define ApplySliceOptimalPhaseShiftGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_multiband_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>
#include "ismrmrd/xml.h"

namespace Gadgetron{

class EXPORTGADGETSMULTIBAND ApplySliceOptimalPhaseShiftGadget :
        public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(ApplySliceOptimalPhaseShiftGadget);

    ApplySliceOptimalPhaseShiftGadget();
    virtual ~ApplySliceOptimalPhaseShiftGadget();

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);      

    void get_multiband_parameters(ISMRMRD::IsmrmrdHeader h);

    arma::ivec map_interleaved_acquisitions(int number_of_slices, bool no_reordering );

    arma::imat get_map_slice_single_band(bool no_reordering);    

    int get_blipped_factor(int numero_de_slice);

    int get_stack_number_from_gt_numerotation(int slice);

    int get_stack_number_from_spatial_numerotation(int slice);

    void deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h);


    /*---------------------*/

    //int transfer_acs_calibration_to_the_next_gadget(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2 , int a);

protected:
    //GADGET_PROPERTY(coil_mask, std::string, "String mask of zeros and ones, e.g. 000111000 indicating which coils to keep", "");



    GADGET_PROPERTY(doOffline, int, "doOffline", 0);
    GADGET_PROPERTY(MB_factor, int, "MB_factor", 2);
    GADGET_PROPERTY(Blipped_CAIPI, int, "Blipped_CAIPI", 4);
    GADGET_PROPERTY(MB_Slice_Inc, int, "MB_Slice_Inc", 2);

    std::vector<size_t> dimensions_;

    unsigned int lNumberOfSlices_;
    unsigned int lNumberOfAverage_;
    unsigned int lNumberOfStacks_;
    unsigned int lNumberOfChannels_;

    unsigned int readout;
    unsigned int encoding;

    unsigned int MB_factor_;
    unsigned int Blipped_CAIPI_;
    unsigned int MB_Slice_Inc_;

    arma::ivec order_of_acquisition_sb;
    arma::ivec order_of_acquisition_mb;

    arma::uvec indice_mb;
    arma::uvec indice_sb;

    arma::imat MapSliceSMS;
    arma::ivec vec_MapSliceSMS;

    //Indice qui sont utilisés pour sauvegarder certains fichiers sur le disque

    std::string str_a;    
    std::string str_s;    
    std::string str_home;

    arma::cx_fvec corrpos_;     // Odd-Even correction -- positive readouts
    arma::cx_fvec corrneg_;     // Odd-Even correction -- negative readouts

    arma::cx_fmat corrpos_mean_;     // Odd-Even correction -- positive readouts
    arma::cx_fmat corrneg_mean_;     // Odd-Even correction -- negative readouts

    arma::cx_fmat corrpos_all_;     // Odd-Even correction -- positive readouts
    arma::cx_fmat corrneg_all_;     // Odd-Even correction -- negative readouts

};
}
#endif /* ApplySliceOptimalPhaseShiftGadget_H_ */

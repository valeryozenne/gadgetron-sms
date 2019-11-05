#ifndef PseudoGenericMultibandRecoGadget_H_
#define PseudoGenericMultibandRecoGadget_H_

#include "GadgetronTimer.h"
#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_multiband_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>
#include "ismrmrd/xml.h"

#include "ImageIOAnalyze.h"

namespace Gadgetron{

class EXPORTGADGETSMULTIBAND PseudoGenericMultibandRecoGadget :
        public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(PseudoGenericMultibandRecoGadget);

    PseudoGenericMultibandRecoGadget();
    virtual ~PseudoGenericMultibandRecoGadget();

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);

    void get_multiband_parameters(ISMRMRD::IsmrmrdHeader h);

    arma::cx_fcube im2col(arma::cx_fcube input);

    arma::cx_fcube im2col_version2(arma::cx_fcube input);

    void calculate_kernel(arma::cx_fcube input); 

    arma::cx_fcube remove_data_if_inplane_grappa(arma::cx_fcube input);     

    void detect_flag(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1);

    arma::cx_fcube shifting_of_multislice_data(arma::cx_fcube input, int PE_shift);

    void set_kernel_parameters(void);

    arma::cx_fcube preliminary_reshape(arma::cx_fmat input);

    void find_encoding_dimension_for_SMS_calculation(void);

    int get_spatial_slice_number(int slice);

    /*---------------------*/

    int transfer_kspace_to_the_next_gadget(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2 , int a);

    void calculate_unfolded_image(arma::cx_fcube input, arma::ivec kernel_slice_position);

    void deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h);


protected:
    //GADGET_PROPERTY(coil_mask, std::string, "String mask of zeros and ones, e.g. 000111000 indicating which coils to keep", "");

    GADGET_PROPERTY_LIMITS(kernel_height, unsigned int, "kernel_height", 5, GadgetPropertyLimitsRange, 1, 9);
    GADGET_PROPERTY_LIMITS(kernel_width, unsigned int, "kernel_width", 5, GadgetPropertyLimitsRange, 1, 9);

    GADGET_PROPERTY(doOffline, int, "doOffline", 0);
    GADGET_PROPERTY(MB_factor, int, "MB_factor", 2);
    GADGET_PROPERTY(Blipped_CAIPI, int, "Blipped_CAIPI", 4);
    GADGET_PROPERTY(MB_Slice_Inc, int, "MB_Slice_Inc", 2);


    Gadgetron::ImageIOAnalyze gt_exporter_;

    std::vector<size_t> dimensions_;

    unsigned int readout;
    unsigned int encoding;
    unsigned int lNumberOfChannels_;
    unsigned int acceFactorE1_;
    unsigned int acceFactorE2_;

    unsigned int MB_factor_;
    unsigned int Blipped_CAIPI_;
    unsigned int MB_Slice_Inc_;

    unsigned int lNumberOfSlices_;
    unsigned int lNumberOfAverage_;
    unsigned int lNumberOfStacks_;



    arma::cx_fcube slice_calibration;
    arma::cx_fcube slice_calibration_reduce;
    arma::cx_fcube slice_calibration_reduce_unblipped;

    arma::cx_fcube folded_image;
    arma::cx_fcube folded_image_reduce;

    arma::cx_fcube unfolded_image_with_blipped;
    arma::cx_fcube unfolded_image_for_output;
    arma::cx_fcube unfolded_image_for_output_all;

    arma::field<arma::cx_fcube> recon_crop;
    arma::field<arma::cx_fcube> recon_prep;
    arma::field<arma::cx_fcube> recon_reshape;

    arma::cx_fcube recon_crop_template;
    arma::cx_fcube recon_prep_template;
    arma::cx_fcube recon_reshape_template;

    bool debug;

    unsigned int last_scan_in_acs;

    arma::fmat matrix_encoding;
    arma::fmat acs_encoding;

    arma::ivec order_of_acquisition_sb;
    arma::ivec order_of_acquisition_mb;

    arma::ivec slice_number_of_acquisition_mb;
    arma::ivec index_of_acquisition_mb;

    arma::uvec indice_mb;
    arma::uvec indice_sb;

    arma::imat MapSliceSMS;

    arma::ivec vec_MapSliceSMS;



    unsigned int e1_size_for_sms_calculation;

    unsigned int flag_encoding_first_in_encoding;
    unsigned int flag_encoding_last_in_encoding;
    unsigned int flag_encoding_first_scan_in_slice;
    unsigned int flag_encoding_last_scan_in_slice;

    arma::cx_fcube kernel;

    arma::cx_fcube kernel_all_slices;

    //Indice qui sont utilisés pour sauvegarder certains fichiers sur le disque
    std::string str_e;
    std::string str_a;
    std::string str_c;
    std::string str_s;
    std::string str_s2;
    std::string str_home;

    unsigned int blocks_per_column;
    unsigned int blocks_per_row;
    unsigned int nb_pixels_per_image;
    unsigned int kernel_size;


    // clock for timing
    Gadgetron::GadgetronTimer gt_timer_local_;
    Gadgetron::GadgetronTimer gt_timer_;


};
}
#endif /* PseudoGenericMultibandRecoGadget_H_ */

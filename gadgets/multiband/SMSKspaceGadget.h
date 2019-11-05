#ifndef EMPTYKSPACEGADGET_H_
#define EMPTYKSPACEGADGET_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_multiband_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

class EXPORTGADGETSMULTIBAND SMSKspaceGadget :
        public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(SMSKspaceGadget);

    SMSKspaceGadget();
    virtual ~SMSKspaceGadget();

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);


    int transfer_kspace_to_the_next_gadget(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2, int grp);


    void save_kspace_data(arma::cx_fcube input, std::string name);

    void find_encoding_dimension_for_SMS_calculation(void);

    void set_kernel_parameters(void);

    void calculate_kernel(arma::cx_fcube input);

    void calculate_unfolded_image(arma::cx_fcube input);

    arma::cx_fcube remove_data_if_inplane_grappa(arma::cx_fcube input);

    arma::cx_fcube preliminary_reshape(arma::cx_fmat input);

    arma::cx_fcube im2col(arma::cx_fcube input);

    arma::cx_fcube im2col_version2(arma::cx_fcube input);

    arma::cx_fcube shifting_of_multislice_data(arma::cx_fcube input, int PE_shift);

    void detect_flag(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1);

protected:

    //GADGET_PROPERTY(coil_mask, std::string, "String mask of zeros and ones, e.g. 000111000 indicating which coils to keep", "");
    GADGET_PROPERTY_LIMITS(kernel_height, unsigned int, "kernel_height", 5, GadgetPropertyLimitsRange, 1, 9);
    GADGET_PROPERTY_LIMITS(kernel_width, unsigned int, "kernel_width", 5, GadgetPropertyLimitsRange, 1, 9);

    std::vector<size_t> dimensions_;

    //arma::cx_fcube parallel_calibration;
    arma::cx_fcube slice_calibration;
    arma::cx_fcube folded_image;
    arma::cx_fcube unfolded_image_without_blipped;
    arma::cx_fcube unfolded_image_without_blipped_inverse;
    arma::cx_fcube unfolded_image_without_blipped_all;
    arma::cx_fcube unfolded_image_without_blipped_inverse_all;
    arma::cx_fcube unfolded_image_with_blipped;

    arma::field<arma::cx_fcube> recon_crop;
    arma::field<arma::cx_fcube> recon_prep;
    arma::field<arma::cx_fcube> recon_reshape;

    arma::cx_fcube recon_crop_template;
    arma::cx_fcube recon_prep_template;
    arma::cx_fcube recon_reshape_template;

    arma::fmat deja_vu;

    bool debug;

    unsigned int slice_de_declenchement;

    unsigned int readout;
    unsigned int encoding;
    unsigned int nbChannels;
    unsigned int acceFactorE1_;
    unsigned int acceFactorE2_;
    unsigned int acceFactorSlice_;

    unsigned int lNumberOfSlices_;
    unsigned int lNumberOfAverage_;

    unsigned int lNumberOfGroup_;

    unsigned int e1_size_for_sms_calculation;

    unsigned int flag_encoding_last_in_encoding;
    unsigned int flag_encoding_first_scan_in_slice;
    unsigned int flag_encoding_last_scan_in_slice;

    arma::ivec flag_is_folded_slice;
    arma::ivec position_folded_slice;
    arma::imat position_debut_fin;

    arma::cx_fcube kernel;

    arma::cx_fcube kernel_all_slices;

    //Indice qui sont utilisés pour sauvegarder certains fichiers sur le disque
    std::string str_e;
    std::string str_c;
    std::string str_s;
    std::string str_home;

    unsigned int blocks_per_column;
    unsigned int blocks_per_row;
    unsigned int nb_pixels_per_image;
    unsigned int kernel_size;
};
}
#endif /* COILREDUCTIONGADGET_H_ */

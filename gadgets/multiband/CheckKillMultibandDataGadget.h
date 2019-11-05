#ifndef CheckMultibandDefoldingGadget_H_
#define CheckMultibandDefoldingGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_multiband_export.h"
#include "hoArmadillo.h"
#include "ismrmrd/xml.h"
#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

class EXPORTGADGETSMULTIBAND CheckMultibandDefoldingGadget :
        public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(CheckMultibandDefoldingGadget);

    CheckMultibandDefoldingGadget();
    virtual ~CheckMultibandDefoldingGadget();

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);

    void get_multiband_parameters(ISMRMRD::IsmrmrdHeader h);

    std::string  Is_Equal(unsigned int a , unsigned int b);

    void deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h);

protected:

    GADGET_PROPERTY(doOffline, int, "doOffline", 0);
    GADGET_PROPERTY(MB_factor, int, "MB_factor", 2);
    GADGET_PROPERTY(Blipped_CAIPI, int, "Blipped_CAIPI", 4);
    GADGET_PROPERTY(MB_Slice_Inc, int, "MB_Slice_Inc", 2);



    std::vector<size_t> dimensions_;

    unsigned int lNumberOfSlices_;
    unsigned int lNumberOfAverage_;
    unsigned int lNumberOfStacks_;
    unsigned int lNumberOfChannels_;


    unsigned int acceFactorE1_;
    unsigned int acceFactorE2_;

    unsigned int MB_factor_;
    unsigned int Blipped_CAIPI_;
    unsigned int MB_Slice_Inc_;

    unsigned int Unknow_2;
    unsigned int Unknow_3;
    unsigned int Unknow_4;

    arma::ivec order_of_acquisition_sb;
    arma::ivec order_of_acquisition_mb;

    arma::ivec slice_number_of_acquisition_mb;
    arma::ivec index_of_acquisition_mb;

    arma::uvec indice_mb;
    arma::uvec indice_sb;

    arma::imat MapSliceSMS;

    arma::ivec vec_MapSliceSMS;

    unsigned int compteur_sb;
    unsigned int compteur_mb;

    arma::ivec real_order_of_acquisition_mb;
    arma::ivec real_order_of_acquisition_sb;

    unsigned int readout;
    unsigned int encoding;

    std::string mon_fichier ;

    //Indice qui sont utilisés pour sauvegarder certains fichiers sur le disque

    std::string str_home;
    std::string str_e;
    std::string str_s;

    arma::fmat deja_vu;
    arma::fmat deja_vu_epi_calib;

};
}
#endif /* WRITESLICECALIBRATIONFLAGS_H_ */

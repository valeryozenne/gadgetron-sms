
#include "GenericReconEigenChannelSMSGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconEigenChannelSMSGadget::GenericReconEigenChannelSMSGadget() : BaseClass()
{
}

GenericReconEigenChannelSMSGadget::~GenericReconEigenChannelSMSGadget()
{
}

int GenericReconEigenChannelSMSGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    ISMRMRD::IsmrmrdHeader h;
    try
    {
        deserialize(mb->rd_ptr(), h);
    }
    catch (...)
    {
        GDEBUG("Error parsing ISMRMRD Header");
    }

    if (!h.acquisitionSystemInformation)
    {
        GDEBUG("acquisitionSystemInformation not found in header. Bailing out");
        return GADGET_FAIL;
    }

    // -------------------------------------------------

    size_t NE = h.encoding.size();
    num_encoding_spaces_ = NE;
    GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

    calib_mode_.resize(NE, ISMRMRD_noacceleration);

    KLT_.resize(NE);

    for (size_t e = 0; e < h.encoding.size(); e++)
    {
        ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
        ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
        ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;

        if (!h.encoding[e].parallelImaging)
        {
            GDEBUG_STREAM("Parallel Imaging section not found in header");
            calib_mode_[e] = ISMRMRD_noacceleration;
        }
        else
        {

            ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;
            std::string calib = *p_imaging.calibrationMode;

            bool separate = (calib.compare("separate") == 0);
            bool embedded = (calib.compare("embedded") == 0);
            bool external = (calib.compare("external") == 0);
            bool interleaved = (calib.compare("interleaved") == 0);
            bool other = (calib.compare("other") == 0);

            calib_mode_[e] = Gadgetron::ISMRMRD_noacceleration;
            if (p_imaging.accelerationFactor.kspace_encoding_step_1 > 1 || p_imaging.accelerationFactor.kspace_encoding_step_2 > 1)
            {
                if (interleaved)
                    calib_mode_[e] = Gadgetron::ISMRMRD_interleaved;
                else if (embedded)
                    calib_mode_[e] = Gadgetron::ISMRMRD_embedded;
                else if (separate)
                    calib_mode_[e] = Gadgetron::ISMRMRD_separate;
                else if (external)
                    calib_mode_[e] = Gadgetron::ISMRMRD_external;
                else if (other)
                    calib_mode_[e] = Gadgetron::ISMRMRD_other;
            }
        }
    }

    return GADGET_OK;
}

int GenericReconEigenChannelSMSGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconEigenChannelSMSGadget::process"); }

    process_called_times_++;

    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
    if (recon_bit_->rbit_.size() > num_encoding_spaces_)
    {
        GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
    }

    // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
    size_t e, n, s, slc;
    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;

        hoNDArray< std::complex<float> >& data_8D = recon_bit_->rbit_[e].data_.data_;

        size_t RO = data_8D.get_size(0);
        size_t E1 = data_8D.get_size(1);
        size_t E2 = data_8D.get_size(2);
        size_t CHA = data_8D.get_size(3);
        size_t MB = data_8D.get_size(4);
        size_t STK = data_8D.get_size(5);
        size_t N = data_8D.get_size(6);
        size_t S = data_8D.get_size(6);

        hoNDArray<std::complex<float> > data_7D(RO, E1, E2, CHA, MB, S, STK);
        reformat_to_7D_for_testing(data_8D, data_7D);

        size_t iRO = data_7D.get_size(0);
        size_t iE1 = data_7D.get_size(1);
        size_t iE2 = data_7D.get_size(2);
        size_t iCHA = data_7D.get_size(3);
        size_t iMB = data_7D.get_size(4);
        size_t iS = data_7D.get_size(5);
        size_t iSTK = data_7D.get_size(6);

        GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N <<  " " << S <<"]");

        if(data_8D.get_number_of_elements()==0)
        {
            m1->release();
            return GADGET_OK;
        }

        // whether it is needed to update coefficients
        bool recompute_coeff = false;

        if ( (KLT_[e].size()!=iSTK) || update_eigen_channel_coefficients.value() )
        {
            recompute_coeff = true;
        }
        else
        {
            if(KLT_[e].size() == iSTK)
            {
                for (slc = 0; slc < iSTK; slc++)
                {

                    if (KLT_[e][slc].size() != iS)
                    {
                        recompute_coeff = true;
                        break;
                    }
                    else
                    {
                        for (s = 0; s < iS; s++)
                        {
                            if (KLT_[e][slc][s].size() != iMB)
                            {
                                recompute_coeff = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        bool average_N = average_all_ref_N.value();
        bool average_S = average_all_ref_S.value();

        if(recompute_coeff)
        {
            if(rbit.ref_)
            {
                hoNDArray< std::complex<float> >& ref_8D = recon_bit_->rbit_[e].ref_->data_;

                size_t RO = ref_8D.get_size(0);
                size_t E1 = ref_8D.get_size(1);
                size_t E2 = ref_8D.get_size(2);
                size_t CHA = ref_8D.get_size(3);
                size_t MB = ref_8D.get_size(4);
                size_t STK = ref_8D.get_size(5);
                size_t N = ref_8D.get_size(6);
                size_t S = ref_8D.get_size(7);

                ref_7D.create(RO, E1, E2, CHA, MB, S, STK);
                reformat_to_7D_for_testing(ref_8D, ref_7D);

                GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N <<  " " << S <<"]");

                size_t iRO = ref_7D.get_size(0);
                size_t iE1 = ref_7D.get_size(1);
                size_t iE2 = ref_7D.get_size(2);
                size_t iCHA = ref_7D.get_size(3);
                size_t iMB = ref_7D.get_size(4);
                size_t iS = ref_7D.get_size(5);
                size_t iSTK = ref_7D.get_size(6);

                //for (size_t a = 0; a < STK; a++) {
                //std::cout << " go KLT " << a << std::endl;
                //std::complex<float> *pIn = &(ref_8D(0, 0, 0, 0, a, 0, 0));
                //hoNDArray<std::complex<float> > tempo_ref(RO, E1, E2, CHA, MB, 1, 1 , pIn);
                //std::cout << " average_N "<<  average_N <<" average_S "<<  average_S  << " calib_mode_[e] "<< calib_mode_[e] <<std::endl;

                // use ref to compute coefficients
                Gadgetron::compute_eigen_channel_coefficients(ref_7D, average_N, average_S,
                                                              false, iMB, iS, upstream_coil_compression_thres.value(), upstream_coil_compression_num_modesKept.value(), KLT_[e]);

                /*std::cout << KLT_[e].size()<< std::endl;    // donc c'est slc
                std::cout << KLT_[e][0].size()<< std::endl;  //  KLT[slc]  donc c'est s
                std::cout << KLT_[e][0][0].size()<< std::endl; //  KLT[slc][s]  donc c'est n
                */
                //}

            }
            else
            {
                // use data to compute coefficients
                Gadgetron::compute_eigen_channel_coefficients(data_7D, average_N, average_S,
                                                              false, iMB, iS, upstream_coil_compression_thres.value(), upstream_coil_compression_num_modesKept.value(), KLT_[e]);
            }

            if (verbose.value())
            {
                hoNDArray< std::complex<float> > E;

                for (slc = 0; slc < iSTK; slc++)
                {
                    for (s = 0; s < iS; s++)
                    {
                        for (n = 0; n < iMB; n++)
                        {
                            KLT_[e][slc][s][n].eigen_value(E);

                            GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - Number of modes kept: " << KLT_[e][slc][s][n].output_length() << " out of " << CHA << "; Eigen value, slc - " << slc << ", S - " << s << ", N - " << n << " : [");

                            for (size_t c = 0; c < E.get_size(0); c++)
                            {
                                GDEBUG_STREAM("        " << E(c));
                            }
                            GDEBUG_STREAM("]");
                        }
                    }
                }
            }
            else
            {
                if(average_N && average_S)
                {
                    for (slc = 0; slc < iSTK; slc++)
                    {
                        GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - Number of modes kept, SLC : " << slc << " - " << KLT_[e][slc][0][0].output_length() << " out of " << CHA);
                    }
                }
                else if(average_N && !average_S)
                {
                    for (slc = 0; slc < iSTK; slc++)
                    {
                        for (s = 0; s < iS; s++)
                        {
                            GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - Number of modes kept, [SLC S] : [" << slc << " " << s << "] - " << KLT_[e][slc][s][0].output_length() << " out of " << CHA);
                        }
                    }
                }
                else if(!average_N && average_S)
                {
                    for (slc = 0; slc < iSTK; slc++)
                    {
                        for (n = 0; n < iMB; n++)
                        {
                            GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - Number of modes kept, [SLC N] : [" << slc << " " << n << "] - " << KLT_[e][slc][0][n].output_length() << " out of " << CHA);
                        }
                    }
                }
                else if(!average_N && !average_S)
                {
                    for (slc = 0; slc < iSTK; slc++)
                    {
                        for (s = 0; s < iS; s++)
                        {
                            for (n = 0; n < iMB; n++)
                            {
                                GDEBUG_STREAM("GenericReconEigenChannelSMSGadget - Number of modes kept, [SLC S N] : [" << slc << " " << s << " " << n << "] - " << KLT_[e][slc][s][n].output_length() << " out of " << CHA);
                            }
                        }
                    }
                }
            }
        }

        if (!debug_folder_full_path_.empty())
        {
            gt_exporter_.export_array_complex(rbit.data_.data_, debug_folder_full_path_ + "data_before_KLT" + os.str());
        }

        // apply KL coefficients
        Gadgetron::apply_eigen_channel_coefficients(KLT_[e], data_7D);
        reformat_to_8D_for_testing(data_7D,rbit.data_.data_);

        if (!debug_folder_full_path_.empty())
        {
            gt_exporter_.export_array_complex(rbit.data_.data_, debug_folder_full_path_ + "data_after_KLT" + os.str());
        }

        if (rbit.ref_)
        {
            if (!debug_folder_full_path_.empty())
            {
                gt_exporter_.export_array_complex(rbit.ref_->data_, debug_folder_full_path_ + "ref_before_KLT" + os.str());
            }

            Gadgetron::apply_eigen_channel_coefficients(KLT_[e], ref_7D);
            reformat_to_8D_for_testing(ref_7D,rbit.ref_->data_);

            if (!debug_folder_full_path_.empty())
            {
                gt_exporter_.export_array_complex(rbit.ref_->data_, debug_folder_full_path_ + "ref_after_KLT" + os.str());
            }
        }
    }

    if (perform_timing.value()) { gt_timer_.stop(); }

    if (this->next()->putq(m1) < 0)
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}


void GenericReconEigenChannelSMSGadget::reformat_to_7D_for_testing(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> > &output)
{
    size_t RO=input.get_size(0);
    size_t E1=input.get_size(1);
    size_t E2=input.get_size(2);
    size_t CHA=input.get_size(3);
    size_t MB=input.get_size(4);
    size_t STK=input.get_size(5);
    size_t N=input.get_size(6);
    size_t S=input.get_size(7);

    //GADGET_CHECK_THROW(lNumberOfSlices_ == STK*MB);

    size_t n, s, a, m;
    int index;

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            for (s = 0; s < S; s++)   {

                for (n = 0; n < N; n++)  {

                    std::complex<float> * in = &(input(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(output(0, 0, 0, 0, m, s, a));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                }
            }
        }
    }
}


void GenericReconEigenChannelSMSGadget::reformat_to_8D_for_testing(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> > &output)
{
    size_t RO=input.get_size(0);
    size_t E1=input.get_size(1);
    size_t E2=input.get_size(2);
    size_t CHA=input.get_size(3);
    size_t MB=input.get_size(4);
    size_t S=input.get_size(5);
    size_t STK=input.get_size(6);

    size_t N=1;
    //GADGET_CHECK_THROW(lNumberOfSlices_ == STK*MB);

    size_t n, s, a, m;
    int index;

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            for (s = 0; s < S; s++)   {

                for (n = 0; n < N; n++)  {

                    std::complex<float> * in = &(input(0, 0, 0, 0, m, s, a));
                    std::complex<float> * out = &(output(0, 0, 0, 0, m, a, n, s));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                }
            }
        }
    }
}




GADGET_FACTORY_DECLARE(GenericReconEigenChannelSMSGadget)
}

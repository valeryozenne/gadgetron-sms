
#include "GenericReconCartesianMultibandPostGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

    GenericReconCartesianMultibandPostGadget::GenericReconCartesianMultibandPostGadget() : BaseClass()
    {
    }

    GenericReconCartesianMultibandPostGadget::~GenericReconCartesianMultibandPostGadget()
    {
    }

    int GenericReconCartesianMultibandPostGadget::process_config(ACE_Message_Block* mb)
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


        number_of_slices = 6;
        MB_factor_ = 2;
        number_of_stacks = number_of_slices / MB_factor_;
        order_of_acquisition_sb = {1, 3, 5, 0, 2, 4};
        order_of_acquisition_mb = {0, 2, 1};
        MapSliceSMS = {{0, 3}, {2, 5}, {1, 4}};

        vec_MapSliceSMS = {0, 3, 2, 5, 1, 4};


        //sort matlab

        indice_mb = {0, 2, 1};
        indice_sb = {3, 0, 4, 1, 5, 2};

        indice_slice_mb = {0, 0, 0};

        // recupere les 18 premiers elements (number_of_stacks) de indice_sb

        for (int i = 0; i < number_of_stacks; i++) {
            indice_slice_mb [i] = indice_sb[i];
        }





        size_t NE = h.encoding.size();
        num_encoding_spaces_ = NE;
        GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

        calib_mode_.resize(NE, ISMRMRD_noacceleration);
        ref_prepared_.resize(NE, false);

        for (size_t e = 0; e < h.encoding.size(); e++)
        {
            ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
            ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
            ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;

            GDEBUG_CONDITION_STREAM(verbose.value(), "---> Encoding space : " << e << " <---");
            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding matrix size: " << e_space.matrixSize.x << " " << e_space.matrixSize.y << " " << e_space.matrixSize.z);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Encoding field_of_view : " << e_space.fieldOfView_mm.x << " " << e_space.fieldOfView_mm.y << " " << e_space.fieldOfView_mm.z);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Recon matrix size : " << r_space.matrixSize.x << " " << r_space.matrixSize.y << " " << r_space.matrixSize.z);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Recon field_of_view :  " << r_space.fieldOfView_mm.x << " " << r_space.fieldOfView_mm.y << " " << r_space.fieldOfView_mm.z);

            if (!h.encoding[e].parallelImaging)
            {
                GDEBUG_STREAM("Parallel Imaging section not found in header");
                calib_mode_[e] = ISMRMRD_noacceleration;
            }
            else
            {

                ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;
                GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE1 is " << p_imaging.accelerationFactor.kspace_encoding_step_1);
                GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE2 is " << p_imaging.accelerationFactor.kspace_encoding_step_2);

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

    int GenericReconCartesianMultibandPostGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {

        if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianMultibandPrepGadget::process"); }

        //process_called_times_++;

        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();


        if (recon_bit_->rbit_.size() > num_encoding_spaces_)
        {
            GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
        }

        // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
        size_t e;
        for (e = 0; e < recon_bit_->rbit_.size(); e++)
        {
            auto & rbit = recon_bit_->rbit_[e];
            std::stringstream os;
            os << "_encoding_" << e;




            if (recon_bit_->rbit_[e].sb_->data_.get_number_of_elements() > 0)
            {

                //modifier le sb

                hoNDArray< std::complex<float> >& data_sb = recon_bit_->rbit_[e].sb_->data_;


                if (!debug_folder_full_path_.empty())
                {
                    //TODO
                    //export_data_for_each_slice(data , "data_avant_remove",   e);

                }


                size_t RO=data_sb.get_size(0);

                size_t E1=data_sb.get_size(1);
                size_t E2=data_sb.get_size(2);
                size_t CHA=data_sb.get_size(3);
                size_t N=data_sb.get_size(4);
                size_t S=data_sb.get_size(5);


                size_t STACK=data_sb.get_size(6);

                size_t SLICE=data_sb.get_size(7);

                hoNDArray< std::complex<float> > data_sb_after;
                data_sb_after.create(RO, E1, E2, CHA, N, S, STACK*SLICE);


                undo_stacks_ordering_to_match_gt_organisation(data_sb, data_sb_after);


                m1->getObjectPtr()->rbit_[e].sb_->data_ = data_sb_after;

                if (!debug_folder_full_path_.empty())
                {

                    //export_data_for_each_stack_and_slice(fid_stack_SB, "data_apres_create", e);
                    //export_data_for_each_slice(data , "data_apres_remove",   e);
                }



                if (this->next()->putq(m1) < 0)
                {
                    GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                    return GADGET_FAIL;
                }

                if (perform_timing.value()) { gt_timer_.stop(); }

                return GADGET_OK;

            }
        }
    }


    void GenericReconCartesianMultibandPostGadget::undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output)
    {
        size_t RO=data.get_size(0);
        size_t E1=data.get_size(1);
        size_t E2=data.get_size(2);
        size_t CHA=data.get_size(3);
        size_t N=data.get_size(4);
        size_t S=data.get_size(5);
        size_t STACK=data.get_size(6);
        size_t SLICE=data.get_size(7);

        size_t n, s, stack, slice;
        int index;

        for (stack = 0; stack < STACK; stack++) {

            for (slice = 0; slice < SLICE; slice++) {

                index = MapSliceSMS[stack][slice];

                for (s = 0; s < S; s++)
                {
                    size_t usedS = s;
                    if (usedS >= S) usedS = S - 1;

                    for (n = 0; n < N; n++)
                    {
                        size_t usedN = n;
                        if (usedN >= N) usedN = N - 1;

                        std::complex<float> * in = &(data(0, 0, 0, 0, n, s, stack, slice));

                        std::complex<float> * out = &(output(0, 0, 0, 0, n, s, indice_sb[index]));

                        for (size_t i = 0; i < RO*E1*E2*CHA; i++)
                        {
                            out[i] = in[i];
                        }
                    }
                }
            }
        }
    }

    GADGET_FACTORY_DECLARE(GenericReconCartesianMultibandPostGadget)
}

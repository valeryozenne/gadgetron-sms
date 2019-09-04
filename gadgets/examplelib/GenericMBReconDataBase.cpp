
#include "GenericMBReconDataBase.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"



namespace Gadgetron {

    GenericMBReconDataBase::GenericMBReconDataBase() : BaseClass()
    {
    }

    GenericMBReconDataBase::~GenericMBReconDataBase()
    {
    }

    int GenericMBReconDataBase::process_config(ACE_Message_Block* mb)
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

/*
        number_of_slices = 36;
        MB_factor_ = 2;
        number_of_stacks = number_of_slices / MB_factor_;
        order_of_acquisition_sb = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34};
        order_of_acquisition_mb = {1, 3, 5, 7, 9, 11, 13, 15, 17, 0, 2, 4, 6, 8, 10, 12, 14, 16};
        MapSliceSMS = {{1, 19}, {3, 21}, {5, 23}, {7, 25}, {9, 27}, {11, 29}, {13, 31}, {15, 33}, {17, 35}, {0, 18}, {2, 20}, {4, 22}, {6, 24}, {8, 26}, {10, 28}, {12, 30}, {14, 32}, {16, 34}};
        vec_MapSliceSMS = {2, 20, 4, 22, 6, 24, 8, 26, 10, 28, 12, 30, 14, 32, 16, 34, 18, 36, 1, 19, 3, 21, 5, 23, 7, 25, 9, 27, 11, 29, 13, 31, 15, 33, 17, 35};


        //sort matlab :

        indice_mb = {10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9};
        indice_sb = {19, 1, 20, 2, 21, 3, 22, 4, 23, 5, 24, 6, 25, 7, 26, 8, 27, 9, 28, 10, 29, 11, 30, 12, 31, 13, 32, 14, 33, 15, 34, 16, 35, 17, 36, 18};

        // si   MB_factor_ = 4;
        //   order_of_acquisition_sb = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34};
        //   order_of_acquisition_mb = { 1 ,    3  ,   5 ,    7  ,   9   , 11 ,  13   , 15   , 17 ,    0  ,   2 ,    4  ,   6 ,    8  ,  10  ,  12  ,  14  ,  16};
        //   MapSliceSMS = {{0, 9, 18, 27}, {2, 11, 20, 29}, {4, 13, 22, 31}, {6, 15, 24, 33}, {8, 17, 26, 35}, {1, 10, 19, 28}, {3, 12, 21, 30}, {5, 14, 23, 32}, {7, 16, 25, 34}};
        //   vec_MapSliceSMS = {1, 10, 19, 28, 3, 12, 21, 30, 5, 14, 23, 32, 7, 16, 25, 34, 9, 18, 27, 36, 2, 11, 20, 29, 4, 13, 22, 31, 6, 15, 24, 33, 8, 17, 26, 35};
        indice_sb = order_of_acquisition_sb;
        indice_mb = order_of_acquisition_mb;
        std::sort(indice_sb.begin(), indice_sb.end());
        std::sort(indice_mb.begin(), indice_mb.end());

        indice_slice_mb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};



        hoNDArray<int> x, r;
        x.create(5);

        x(0) = 5;
        x(1) = 2;
        x(2) = 3;
        x(3) = 4;
        x(4) = 1;

        std::vector<size_t> ind;
        bool isascending = true;
        Gadgetron::sort(x, r, ind, isascending);


        for (int i = 0; i < number_of_stacks; i++) {
         std::cout << " i "<< i << " index " <<   ind [i] <<  std::endl;
        }*/


        //void const hoNDArray<T>& x, hoNDArray<T>& r, std::vector<size_t>& ind, bool isascending);



        /* recupere les 18 premiers elements (number_of_stacks) de indice_sb

        for (int i = 0; i < number_of_stacks; i++) {
            indice_slice_mb [i] = indice_sb[i];
        }
*/



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

    int GenericMBReconDataBase::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
    {


        if (perform_timing.value()) { gt_timer_.start("GenericMBReconDataBase::process"); }

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

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

            if (!debug_folder_full_path_.empty())
            {
                gt_exporter_.export_array_complex(recon_bit_->rbit_[e].data_.data_, debug_folder_full_path_ + "data_debut" + os.str());
            }



            if (!debug_folder_full_path_.empty())
            {
                gt_exporter_.export_array_complex(recon_bit_->rbit_[e].data_.data_, debug_folder_full_path_ + "data_fin" + os.str());
            }




            /*

                //autre méthode pas de changement de valeur just eune copie
                // memcpy(out, in, sizeof(std::complex<float>)*RO*E1*E2*CHA*N*S);


            }

*/

        if (this->next()->putq(m1) < 0)
        {
            GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
            return GADGET_FAIL;
        }

        if (perform_timing.value()) { gt_timer_.stop(); }

        return GADGET_OK;

        }
    }









    GADGET_FACTORY_DECLARE(GenericMBReconDataBase)
}



#include "GenericReconCartesianMultibandPrepGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconCartesianMultibandPrepGadget::GenericReconCartesianMultibandPrepGadget() : BaseClass()
{
}

GenericReconCartesianMultibandPrepGadget::~GenericReconCartesianMultibandPrepGadget()
{
}

int GenericReconCartesianMultibandPrepGadget::process_config(ACE_Message_Block* mb)
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

    /*number_of_slices = 36;
    MB_factor_ = 2;
    number_of_stacks = number_of_slices / MB_factor_;
    order_of_acquisition_sb = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34};
    order_of_acquisition_mb = {1, 3, 5, 7, 9, 11, 13, 15, 17, 0, 2, 4, 6, 8, 10, 12, 14, 16};
    MapSliceSMS = {{1, 19}, {3, 21}, {5, 23}, {7, 25}, {9, 27}, {11, 29}, {13, 31}, {15, 33}, {17, 35}, {0, 18}, {2, 20}, {4, 22}, {6, 24}, {8, 26}, {10, 28}, {12, 30}, {14, 32}, {16, 34}};

    vec_MapSliceSMS = {1, 19, 3, 21, 5, 23, 7, 25, 9, 27, 11, 29, 13, 31, 15, 33, 17, 35, 0, 18, 2, 20, 4, 22, 6, 24, 8, 26, 10, 28, 12, 30, 14, 32, 16, 34};


    //sort matlab

    indice_mb = {9, 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8};
    indice_sb = {18, 0, 19, 1, 20, 2, 21, 3, 22, 4, 23, 5, 24, 6, 25, 7, 26, 8, 27, 9, 28, 10, 29, 11, 30, 12, 31, 13, 32, 14, 33, 15, 34, 16, 35, 17};

    indice_slice_mb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // recupere les 18 premiers elements (number_of_stacks) de indice_sb

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

int GenericReconCartesianMultibandPrepGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
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


         if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
         {

             hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

             if (!debug_folder_full_path_.empty())
             {
                 //export_data_for_each_slice(data , "data_avant_remove",   e);
             }

             // reorganize data with special index

             remove_extra_dimension(data);


             if (!debug_folder_full_path_.empty())
             {
                 //export_data_for_each_slice(data , "data_apres_remove",   e);

             }


          }


         if (recon_bit_->rbit_[e].sb_)
         {


             hoNDArray< std::complex<float> >& data_sb = recon_bit_->rbit_[e].sb_->data_;

             size_t RO=data_sb.get_size(0);
             size_t E1=data_sb.get_size(1);
             size_t E2=data_sb.get_size(2);
             size_t CHA=data_sb.get_size(3);
             size_t N=data_sb.get_size(4);
             size_t S=data_sb.get_size(5);
             size_t SLC=data_sb.get_size(6);


             hoNDArray< std::complex<float> > fid_stack_SB;

             fid_stack_SB.create(RO, E1, E2, CHA, N, S, number_of_stacks, MB_factor_);

             create_stacks_of_slices(data_sb, fid_stack_SB);

             m1->getObjectPtr()->rbit_[e].sb_->data_ = fid_stack_SB;

             /*RO=data_sb.get_size(0);
             E1=data_sb.get_size(1);
             E2=data_sb.get_size(2);
             CHA=data_sb.get_size(3);
             N=data_sb.get_size(4);
             S=data_sb.get_size(5);
             SLC=m1->getObjectPtr()->rbit_[e].sb_->data_.get_size(6);
             size_t stackz = m1->getObjectPtr()->rbit_[e].sb_->data_.get_size(7);
*/

             //to export data
             if (!debug_folder_full_path_.empty())
             {
                 //export_data_for_each_stack_and_slice(data_sb, "data_apres_create", e);
                // export_data_for_each_slice(data , "data_apres_remove",   e);
             }


         }


         //if (!debug_folder_full_path_.empty())
        //{
        //gt_exporter_.export_array_complex(recon_bit_->rbit_[e].data_.data_, debug_folder_full_path_ + "data_apres_remove" + os.str());
        //}



        if (this->next()->putq(m1) < 0)
        {
            GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
            return GADGET_FAIL;
        }

        if (perform_timing.value()) { gt_timer_.stop(); }

        return GADGET_OK;

    }
}

void GenericReconCartesianMultibandPrepGadget::export_data_for_each_stack_and_slice(hoNDArray< std::complex<float> >& data , std::string filename,  size_t e)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLICE = data.get_size(6);
    size_t STACK = data.get_size(7);

    std::stringstream os;
    os << "_encoding_" << e;

    std::cout << "Saving incoming data array : [RO, E1, E2, CHA, N, S, SLICE, STACK] : " << RO << " " << E1 << " " << E2 << " " << CHA << " " << N  << " " << S << " " << SLICE << STACK << std::endl;

    hoNDArray< std::complex<float> > DATA_4D;
    DATA_4D.create(RO, E1, E2, CHA);

    size_t n, s, slice, stack;

    for (slice = 0; slice < SLICE; slice++) {

        for (stack = 0; stack < STACK; stack++) {

            for (s = 0; s < S; s++)
            {
                size_t usedS = s;
                if (usedS >= S) usedS = S - 1;

                for (n = 0; n < N; n++)
                {
                    size_t usedN = n;
                    if (usedN >= N) usedN = N - 1;


                    {

                        std::stringstream oslc;
                        oslc << "_stack_and_slice_" << slice << stack;

                        std::complex<float>* in = &(data(0, 0, 0, 0, n, s, slice, stack));

                        memcpy(DATA_4D.begin(), in, sizeof( std::complex<float>)*RO*E1*E2*CHA);

                        gt_exporter_.export_array_complex( DATA_4D  , debug_folder_full_path_ + filename + oslc.str() + os.str() );

                    }
                }
            }
        }

    }
}




void GenericReconCartesianMultibandPrepGadget::export_data_for_each_slice(hoNDArray< std::complex<float> >& data , std::string filename,  size_t e)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    std::stringstream os;
    os << "_encoding_" << e;

    std::cout << "Saving incoming data array : [RO, E1, E2, CHA, N, S, SLC] : " << RO << " " << E1 << " " << E2 << " " << CHA << " " << N  << " " << S << " " << SLC << std::endl;

    hoNDArray< std::complex<float> > DATA_4D;
    DATA_4D.create(RO, E1, E2, CHA);

    size_t n, s, slc;

    for (s = 0; s < S; s++)
    {
        size_t usedS = s;
        if (usedS >= S) usedS = S - 1;

        for (n = 0; n < N; n++)
        {
            size_t usedN = n;
            if (usedN >= N) usedN = N - 1;


            for (slc = 0; slc < SLC; slc++)    {

                std::stringstream oslc;
                oslc << "_slice_" << slc;

                std::complex<float>* in = &(data(0, 0, 0, 0, n, s, slc));

                memcpy(DATA_4D.begin(), in, sizeof( std::complex<float>)*RO*E1*E2*CHA);

                gt_exporter_.export_array_complex( DATA_4D  , debug_folder_full_path_ + filename + oslc.str() + os.str() );

            }
        }
    }
}



void GenericReconCartesianMultibandPrepGadget::remove_extra_dimension(hoNDArray< std::complex<float> >& data)
{
    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    hoNDArray< std::complex<float> > FID_MB;
    FID_MB.create(RO, E1, E2, CHA, N, S, number_of_stacks);

    size_t nb_elements_multiband = data.get_number_of_elements()/MB_factor_;

    size_t index_in;
    size_t index_out;

    size_t n, s;
    for (int a = 0; a < number_of_stacks; a++)
    {
        index_in=indice_slice_mb[a];
        index_out=indice_mb[a];

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index_in));
                std::complex<float> * out = &(FID_MB(0, 0, 0, 0, n, s, index_out));

                for (size_t i = 0; i < RO*E1*E2*CHA; i++)
                {
                    out[i] = in[i];
                }
            }
        }
    }

    data = FID_MB;

}


void GenericReconCartesianMultibandPrepGadget::reorganize_data(hoNDArray< std::complex<float> >& data, std::vector<int> indice)
{
    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    hoNDArray< std::complex<float> > new_data;
    new_data.create(RO,E1, E2, CHA, N, S, SLC);

    size_t n, s;

    for (int i = 0; i < SLC; i++) {

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice[i]));
                std::complex<float> * out = &(new_data(0, 0, 0, 0, n, s, i));

                for (size_t i = 0; i < RO*E1*E2*CHA; i++)
                {
                    out[i] = in[i];
                }
            }
        }
    }
    data = new_data;

}


//sur les données single band
void GenericReconCartesianMultibandPrepGadget::create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& fid_stack_SB)
{

    reorganize_data(data, indice_sb);

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    size_t n, s;
    int index, m;

    // copy of the data in the 8D array

    for (int a = 0; a < number_of_stacks; a++) {

        for (m = 0; m < MB_factor_; m++) {

            index = MapSliceSMS[a][m];


            for (s = 0; s < S; s++)
            {
                size_t usedS = s;
                if (usedS >= S) usedS = S - 1;

                for (n = 0; n < N; n++)
                {
                    size_t usedN = n;
                    if (usedN >= N) usedN = N - 1;

                    std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index));
                    std::complex<float> * out = &(fid_stack_SB(0, 0, 0, 0, n, s, a, m));

                    for (size_t i = 0; i < RO*E1*E2*CHA; i++)
                    {
                        out[i] = in[i];
                    }
                }
            }
        }
    }

}



GADGET_FACTORY_DECLARE(GenericReconCartesianMultibandPrepGadget)
}


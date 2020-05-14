
#include "GenericReconSaveAndLoadRefGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconSaveAndLoadRefGadget::GenericReconSaveAndLoadRefGadget() : BaseClass()
{
}

GenericReconSaveAndLoadRefGadget::~GenericReconSaveAndLoadRefGadget()
{
}

int GenericReconSaveAndLoadRefGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}

int GenericReconSaveAndLoadRefGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconSaveAndLoadRefGadget::process"); }

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

        if (recon_bit_->rbit_[e].ref_)
        {
            // std::cout << " je suis la structure qui contient les données acs" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].ref_->data_;
            hoNDArray< ISMRMRD::AcquisitionHeader > headers_ = recon_bit_->rbit_[e].ref_->headers_;
            ISMRMRD::AcquisitionHeader & curr_header = headers_(0, 0, 0, 0, 0);
            std::cout << " repetition" << curr_header.idx.repetition << std::endl;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            GDEBUG_STREAM("GenericReconSaveAndLoadRefGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

            size_t start_E1(0), end_E1(0);
            auto t = Gadgetron::detect_sampled_region_E1(data);
            start_E1 = std::get<0>(t);
            end_E1 = std::get<1>(t);

            GDEBUG_STREAM("GenericReconSaveAndLoadRefGadget detect_sampled_region_E1   "<<  start_E1 << " "<<   end_E1);

            if (save.value()) {
                if (!debug_folder_full_path_.empty()) {

                    std::stringstream as;
                    as << "_acq_" << save_number.value();

                    gt_exporter_.export_array_complex(recon_bit_->rbit_[e].ref_->data_,
                                                      debug_folder_full_path_ + "multiple_ref" + os.str()+ as.str());

                    GDEBUG_STREAM("GenericReconSaveAndLoadRefGadget saved   "<< save_number.value()  << " "<<   debug_folder_full_path_+ "multiple_ref" + os.str()+ as.str()+ "_REAL");
                }
            }

            if (load.value()) {

                hoNDArray< std::complex<float> > new_data(data);

                new_data.fill(0);

                for (int ll = 0; ll <= load_number.value(); ll++)    {


                    std::stringstream as;
                    as << "_acq_" << ll;

                    hoNDArray< std::complex<float> > data_loaded;

                    gt_exporter_.import_array_complex(data_loaded,
                                                      debug_folder_full_path_ + "multiple_ref" + os.str()+ as.str()+ "_REAL",
                                                      debug_folder_full_path_ + "multiple_ref" + os.str()+ as.str()+ "_IMAG");


                    GDEBUG_STREAM("GenericReconSaveAndLoadRefGadget loaded   "<< ll << " "<<   debug_folder_full_path_+"multiple_ref" + os.str()+ as.str()+ "_REAL");
                    std::cout << "GenericReconSaveAndLoadRefGadget  "<< new_data(RO/2,start_E1,0,0,0,0,0)<< " "<<  data(RO/2,start_E1,0,0,0,0,0) << std::endl;

                    Gadgetron::add(new_data, data_loaded, new_data);
                }

                std::cout << "GenericReconSaveAndLoadRefGadget  add "<< new_data(RO/2,start_E1,0,0,0,0,0)<< " "<<  data(RO/2,start_E1,0,0,0,0,0) << std::endl;

                Gadgetron::scal((float)(1.0 / (load_number.value()+1)), new_data);

                std::cout << "GenericReconSaveAndLoadRefGadget scal "<< new_data(RO/2,start_E1,0,0,0,0,0)<< " "<<  data(RO/2,start_E1,0,0,0,0,0) << std::endl;


                recon_bit_->rbit_[e].ref_->data_=new_data;


                //std::cout << "GenericReconSaveAndLoadRefGadget  "<< new_data(RO/2,E1/2,0,0,0,0,SLC-1)<< " "<< data(RO/2,E1/2,0,0,0,0,SLC-1)  << std::endl;
            }

        }



        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;
            hoNDArray< ISMRMRD::AcquisitionHeader > headers_ = recon_bit_->rbit_[e].data_.headers_;
            ISMRMRD::AcquisitionHeader & curr_header = headers_(0, 0, 0, 0, 0);
            std::cout << " repetition" << curr_header.idx.repetition << std::endl;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            //GDEBUG_STREAM("GenericReconSaveAndLoadRefGadget - incoming data array data: [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");


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

GADGET_FACTORY_DECLARE(GenericReconSaveAndLoadRefGadget)
}

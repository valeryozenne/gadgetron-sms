//SegmentedAquisitionGadget.cpp

#include "SegmentedAquisitionGadget.h"
#include "ismrmrd/xml.h"
#include "hoNDArray_fileio.h"

namespace Gadgetron {

    SegmentedAquisitionGadget::SegmentedAquisitionGadget() : BaseClass()
    {
    }

    SegmentedAquisitionGadget::~SegmentedAquisitionGadget()
    {
    }

    int SegmentedAquisitionGadget::process_config(ACE_Message_Block* mb)
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

        size_t NE = h.encoding.size();
        num_encoding_spaces_ = NE;
        GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);
        size_t e;
        
        ref_data = false;
        return GADGET_OK;
    }

    int SegmentedAquisitionGadget::process(GadgetContainerMessage<IsmrmrdReconData>* m1)
    {
        IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();

        if (recon_bit_->rbit_.size() > num_encoding_spaces_)
        {
            GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
        }

        // for every encoding space
        for (size_t e = 0; e < recon_bit_->rbit_.size(); e++)
        {
            //General setup and Checking of encoding space
            auto & rbit = recon_bit_->rbit_[e];
            std::stringstream os;
            os << "_encoding_" << e;
            GDEBUG_CONDITION_STREAM(verbose.value(), "Calling " << process_called_times_ << " , encoding space : " << e);
            GDEBUG_CONDITION_STREAM(verbose.value(), "======================================================================");

            //general variables
            hoNDArray< std::complex<float> >& data = rbit.data_.data_;
            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);
            size_t x, y, cha;
            std::vector<size_t> ind(11);
            //here starts the fun
            if (Sampling_Scheme.value() == "Basic")
            {
                // find center of header region and check if in one of the three central headers the reference shot flag is on
                int center = rbit.data_.headers_.get_size(0)/2 ;
                bool ref_shot_flag = (rbit.data_.headers_[center-1].isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1) || rbit.data_.headers_[center+1].isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1) || rbit.data_.headers_[center].isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1));
                bool dynamic_shot_flag = (rbit.data_.headers_[center-1].isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2) || rbit.data_.headers_[center+1].isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2) || rbit.data_.headers_[center].isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2));

                if (ref_shot_flag == 1)
                {
                    //fill refernce buffer
                    arma::cx_fmat adata = &as_arma_matrix(m1->getObjectPtr());

                    ref_shot_buf_data.create(rbit.data_.data_.get_dimensions());
                    ref_shot_buf_data.clear();
                    ref_shot_buf_data.copyFrom(rbit.data_.data_);
                    ref_data = true;
                    //ref_shot_buf_head.create(rbit.data_.headers_.get_dimensions());  implement header coying aswell
                    //ref_shot_buf_head.copyFrom(rbit.data_.headers_);
                    GDEBUG_STREAM("Reference ReconBit is passed");
                }

                else if (dynamic_shot_flag == 1)
                {
                    GDEBUG_STREAM("Dynamic Shot");
                    
                    if (ref_data == false)
                    {
                        //need to hold data
                        GDEBUG_STREAM("No Reference Data yet");
                    }
                    
                    else if (ref_data == true)
                    {

                        GDEBUG_STREAM("Dynamic");
                        
                        //to do sending the holded or altered data to the next gadget
                        //replacement
                        for (x = 0; x < RO; x++)
                        {
                            for (y = 0; y < E1; y++)
                            {
                                for (cha=0; cha < CHA; cha++)
                                {
                                    ind[0] = 0 ;
                                    ind[1] = x ;
                                    ind[2] = y ;
                                    ind[3] = cha ;
                                    ind[4] = 0 ;
                                    ind[5] = 0 ;
                                    ind[6] = 0 ;
                                    ind[7] = 0 ;
                                    ind[8] = 0 ;
                                    ind[9] = 0 ;
                                    ind[10] = 0 ;

                                    size_t array_index_dynamic = recon_bit_->rbit_[e].data_.data_.calculate_offset(ind);
                                    size_t array_index_ref = ref_shot_buf_data.calculate_offset(ind);

                                    std::complex<float>* pData1 = &(recon_bit_->rbit_[e].data_.data_(array_index_dynamic));
                                    std::complex<float>* pData2 = &(ref_shot_buf_data(array_index_ref));
                                    std::complex<float> foo;
                                    GDEBUG_STREAM("Basic dynamic_shot_flag" << *pData1);
                                    if ((*pData1 == foo) && (*pData2 != foo))
                                    {
                                        *pData1 = *pData2;
                                    }
                                    GDEBUG_STREAM("Test Ich Bin Ein Test");
                                    GDEBUG_STREAM("Basic dynamic_shot_flag" << *pData1);
                                }
                            }
                            
                        }
                        
                        
                    }
                    
                    else
                    {
                        GDEBUG_STREAM("Something with the reference data is off");
                    }
                }
                else
                {
                    GDEBUG_STREAM("No user flag for segmentation was set");
                }           
            }
            else if (Sampling_Scheme.value() == "RIGID")
            {
                GDEBUG_STREAM("RIGID");
            }
        }
        if (this->next()->putq(m1) < 0)
        {
            GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
            return GADGET_FAIL;
        }
        return GADGET_OK;
    }
    GADGET_FACTORY_DECLARE(SegmentedAquisitionGadget)
}

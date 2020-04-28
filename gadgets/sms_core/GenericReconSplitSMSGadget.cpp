
#include "GenericReconSplitSMSGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconSplitSMSGadget::GenericReconSplitSMSGadget() : BaseClass()
{
}

GenericReconSplitSMSGadget::~GenericReconSplitSMSGadget()
{
}

int GenericReconSplitSMSGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}

int Gadgetron::GenericReconSplitSMSGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconSplitSMSGadget::process"); }

    process_called_times_++;

    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();

    if (recon_bit_->rbit_.size() > num_encoding_spaces_)
    {
        GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
    }

    // for every encoding space, split the recon_bit_->rbit_[e].data_
    // size_t e, n, s, slc;
    //for (e = 0; e < recon_bit_->rbit_.size(); e++)
    //{
    size_t e=0;



    auto & rbit = recon_bit_->rbit_[e];
    std::stringstream os;
    os << "_encoding_" << e;

    if (rbit.data_.data_.get_number_of_elements() > 0)
    {
        // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

        hoNDArray< std::complex<float> >& data =rbit.data_.data_;
        hoNDArray< ISMRMRD::AcquisitionHeader > headers_ = rbit.data_.headers_;

        size_t RO = data.get_size(0);
        size_t E1 = data.get_size(1);
        size_t E2 = data.get_size(2);
        size_t CHA = data.get_size(3);
        size_t N = data.get_size(4);
        size_t S = data.get_size(5);
        size_t SLC = data.get_size(6);

        GDEBUG_STREAM("GenericCheckSplitSizeGadget - incoming data array data: [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

        if (N==2)
        {
            GDEBUG("GenericCheckSizeGadget : detecting single band and multiband data, it should be the first repetition\n");

            size_t ii;
            for (ii=0; ii<rbit.data_.headers_.get_number_of_elements(); ii++)
            {
                if( rbit.data_.headers_(ii).idx.repetition>0 )
                {
                    GERROR_STREAM("After checking, it is not the first repetition, something went wrong ... ");
                    return GADGET_FAIL;
                }
            }

            GDEBUG("After checking, it is the first repetition\n");

            size_t new_N=1;

            std::cout << "allocation" << std::endl;
            headers_mb.create(E1, E2, new_N, S , SLC );
            mb.create(RO, E1, E2, CHA, new_N, S , SLC );

            headers_sb.create(E1, E2, new_N, S , SLC );
            sb.create(RO, E1, E2, CHA, new_N, S , SLC );

            //TODO instead of using sb and headers_sb , it should be better to create something like  "recon_bit_sb->rbit_[e] "
            std::cout << "ok" << std::endl;

            if (use_omp.value()==true)
            {
                if (perform_timing.value()) { gt_timer_.start("GenericReconSplitSMSGadget::extract_sb_and_mb_from_data_open"); }
                extract_sb_and_mb_from_data_open( rbit, sb,  mb, headers_sb,  headers_mb);
                if (perform_timing.value()) { gt_timer_.stop(); }
            }
            else
            {
                if (perform_timing.value()) { gt_timer_.start("GenericReconSplitSMSGadget::extract_sb_and_mb_from_data"); }
                extract_sb_and_mb_from_data( rbit, sb,  mb, headers_sb,  headers_mb);
                if (perform_timing.value()) { gt_timer_.stop(); }
            }

            std::cout << "extraction ok " << std::endl;

            //Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m2 = new GadgetContainerMessage< IsmrmrdReconData >();
            //*m2->getObjectPtr() = *m1->getObjectPtr();
            Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m2 = m1->duplicate();

            m1->getObjectPtr()->rbit_[e].data_.data_=sb;
            m1->getObjectPtr()->rbit_[e].data_.headers_=headers_sb;

            std::cout << "prep putq sb m1 ok" << std::endl;

            if (this->next()->putq(m1) < 0)
            {
                GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                return GADGET_FAIL;
            }

            std::cout << "putq sb m1 " << std::endl;

            m2->getObjectPtr()->rbit_[e].data_.data_=mb;
            m2->getObjectPtr()->rbit_[e].data_.headers_=headers_mb;

            //m2->getObjectPtr()->rbit_[e].ref_->clear();
            m2->getObjectPtr()->rbit_[e].ref_ = Core::none;

            std::cout << "prep putq  m2" << std::endl;

            if (this->next()->putq(m2) < 0)
            {
                GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                return GADGET_FAIL;
            }

            std::cout << "putq m2 ok " << std::endl;

        }
        else
        {

            std::cout << "prep putq mb m1" << std::endl;

            //m1->getObjectPtr()->rbit_[e].data_.headers_=headers_mb;

            size_t hE1 = m1->getObjectPtr()->rbit_[e].data_.headers_.get_size(0);
            size_t hE2 = m1->getObjectPtr()->rbit_[e].data_.headers_.get_size(1);
            size_t hN = m1->getObjectPtr()->rbit_[e].data_.headers_.get_size(2);
            size_t hS = m1->getObjectPtr()->rbit_[e].data_.headers_.get_size(3);
            size_t hSLC = m1->getObjectPtr()->rbit_[e].data_.headers_.get_size(4);

            GDEBUG_STREAM("GenericSMSSplitGadget - incoming mb headers_ array : [E1, E2, N, S, LOC] - [" << hE1 << " " << hE2 << " " << hN << " " << hS << " " << hSLC << "]");

            if (this->next()->putq(m1) < 0)
            {
                GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
                return GADGET_FAIL;
            }

            std::cout << "putq mb m1 ok" << std::endl;

        }
    }
    // }

    if (perform_timing.value()) { gt_timer_.stop(); }

    return GADGET_OK;
}



//sur les données single band
void GenericReconSplitSMSGadget::extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb)
{
    //TODO instead of creating a new sb and sb_header i t would be easier to create a new reconbit

    hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit.data_.headers_;  //5D, fixed order [E1, E2, N, S, LOC]

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    size_t hE1=headers_.get_size(0);
    size_t hE2=headers_.get_size(1);
    size_t hN=headers_.get_size(2);
    size_t hS=headers_.get_size(3);
    size_t hSLC=headers_.get_size(4);

    GDEBUG_STREAM("GenericSMSSplitGadget - incoming headers_ array : [E1, E2, N, S, LOC] - [" << hE1 << " " << hE2 << " " << hN << " " << hS << " " << hSLC << "]");

    if (N!=2)
    {
        GERROR_STREAM("size(N) should be equal to 2 ");
    }

    size_t n, s, slc;

    for (slc = 0; slc < SLC; slc++)
    {
        for (s = 0; s < S; s++)
        {
            for (n = 0; n < N; n++)
            {
                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, slc));
                //GDEBUG_stream("data: " << data);
                std::complex<float> * out_sb = &(sb(0, 0, 0, 0, 0, s, slc));
                std::complex<float> * out_mb = &(mb(0, 0, 0, 0, 0, s, slc));

                ISMRMRD::AcquisitionHeader *in_h=&(headers_(0,0,n,s,slc));
                ISMRMRD::AcquisitionHeader *out_h_sb=&(h_sb(0,0,0,s,slc));
                ISMRMRD::AcquisitionHeader *out_h_mb=&(h_mb(0,0,0,s,slc));

                if (n==1)
                {
                    memcpy(out_sb , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                    memcpy(out_h_sb , in_h, sizeof(ISMRMRD::AcquisitionHeader)*E1*E2);
                }
                else
                {
                    memcpy(out_mb , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                    memcpy(out_h_mb , in_h, sizeof(ISMRMRD::AcquisitionHeader)*E1*E2);
                }
            }
        }
    }



    // how to put for instance
    // hoNDArray< std::complex<float> > sb;
    // and hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb;
    // into a new_recon_bit.data_,  what is the contructor of new_recon_bit.data_, assuming only one recon_bit ? I guess it also require some memory allocation;
    // in order to have new_recon_bit.data_.data=sb;
    // and new_recon_bit.data_.headers_=h_sb;

}



//sur les données single band
void GenericReconSplitSMSGadget::extract_sb_and_mb_from_data_open(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb)
{
    //TODO instead of creating a new sb and sb_header i t would be easier to create a new reconbit

    hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit.data_.headers_;  //5D, fixed order [E1, E2, N, S, LOC]

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    size_t hE1=headers_.get_size(0);
    size_t hE2=headers_.get_size(1);
    size_t hN=headers_.get_size(2);
    size_t hS=headers_.get_size(3);
    size_t hSLC=headers_.get_size(4);

    GDEBUG_STREAM("GenericSMSSplitGadget - incoming headers_ array : [E1, E2, N, S, LOC] - [" << hE1 << " " << hE2 << " " << hN << " " << hS << " " << hSLC << "]");

    if (N!=2)
    {
        GERROR_STREAM("size(N) should be equal to 2 ");
    }


    long long num = N * S * SLC;
    long long ii;

    // only allow this for loop openmp if num>1 and 2D recon
#pragma omp parallel for default(none) private(ii) shared(num, S,  N,  RO, E1, E2, CHA,data, sb, mb, headers_, h_sb, h_mb  ) if(num>1)
    for (ii = 0; ii < num; ii++) {
        size_t slc = ii / (N * S);
        size_t s = (ii - slc * N * S) / (N);
        size_t n = ii - slc * N * S - s * N;


        //for (slc = 0; slc < SLC; slc++)   {
        //  for (s = 0; s < S; s++)       {
        //    for (n = 0; n < N; n++)          {

        std::complex<float> * in = &(data(0, 0, 0, 0, n, s, slc));
        std::complex<float> * out_sb = &(sb(0, 0, 0, 0, 0, s, slc));
        std::complex<float> * out_mb = &(mb(0, 0, 0, 0, 0, s, slc));

        ISMRMRD::AcquisitionHeader *in_h=&(headers_(0,0,n,s,slc));
        ISMRMRD::AcquisitionHeader *out_h_sb=&(h_sb(0,0,0,s,slc));
        ISMRMRD::AcquisitionHeader *out_h_mb=&(h_mb(0,0,0,s,slc));

        if (n==1)
        {
            memcpy(out_sb , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
            memcpy(out_h_sb , in_h, sizeof(ISMRMRD::AcquisitionHeader)*E1*E2);
        }
        else
        {
            memcpy(out_mb , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
            memcpy(out_h_mb , in_h, sizeof(ISMRMRD::AcquisitionHeader)*E1*E2);
        }
        //  }
        //}
    }



    // how to put for instance
    // hoNDArray< std::complex<float> > sb;
    // and hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb;
    // into a new_recon_bit.data_,  what is the contructor of new_recon_bit.data_, assuming only one recon_bit ? I guess it also require some memory allocation;
    // in order to have new_recon_bit.data_.data=sb;
    // and new_recon_bit.data_.headers_=h_sb;

}



GADGET_FACTORY_DECLARE(GenericReconSplitSMSGadget)
}

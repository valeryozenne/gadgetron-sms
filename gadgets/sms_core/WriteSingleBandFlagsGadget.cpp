#include "WriteSingleBandFlagsGadget.h"

namespace Gadgetron{


WriteSingleBandFlagsGadget::WriteSingleBandFlagsGadget() {
}

WriteSingleBandFlagsGadget::~WriteSingleBandFlagsGadget() {
}

int WriteSingleBandFlagsGadget::process_config(ACE_Message_Block *mb)
{
  ISMRMRD::IsmrmrdHeader h;
  ISMRMRD::deserialize(mb->rd_ptr(),h);





      size_t NE = h.encoding.size();
      num_encoding_spaces_ = NE;
      GDEBUG_STREAM("Number of encoding spaces: " << NE);

      if (h.encoding.size() != 1)
      {
          GDEBUG("Number of encoding spaces: %d\n", h.encoding.size());
      }

      size_t e = 0;
      ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
      ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
      ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;

      meas_size_idx_.kspace_encode_step_1 = (uint16_t)e_space.matrixSize.y ;
      meas_size_idx_.set = e_limits.set ? e_limits.set->maximum - e_limits.set->minimum + 1 : 1;
      meas_size_idx_.phase = e_limits.phase ? e_limits.phase->maximum - e_limits.phase->minimum + 1 : 1;
      meas_size_idx_.kspace_encode_step_2 = (uint16_t)e_space.matrixSize.z ;
      meas_size_idx_.contrast = e_limits.contrast ? e_limits.contrast->maximum - e_limits.contrast->minimum + 1 : 1;
      meas_size_idx_.slice = e_limits.slice ? e_limits.slice->maximum - e_limits.slice->minimum + 1 : 1;
      meas_size_idx_.repetition = e_limits.repetition ? e_limits.repetition->maximum - e_limits.repetition->minimum + 1 : 1;
      meas_size_idx_.average = e_limits.average ? e_limits.average->maximum - e_limits.average->minimum + 1 : 1;
      meas_size_idx_.segment = e_limits.segment ? e_limits.segment->maximum - e_limits.segment->minimum + 1 : 1;

      std::cout << " val: meas_size_idx_.kspace_encode_step_1 " << meas_size_idx_.kspace_encode_step_1 << std::endl;
      std::cout << " val: meas_size_idx_.kspace_encode_step_2 " << meas_size_idx_.kspace_encode_step_2 << std::endl;
      std::cout << " val: meas_size_idx_.contrast " << meas_size_idx_.contrast << std::endl;
      std::cout << " val: meas_size_idx_.slice " << meas_size_idx_.slice << std::endl;
      std::cout << " val: meas_size_idx_.repetition " << meas_size_idx_.repetition << std::endl;
      std::cout << " val: meas_size_idx_.average " << meas_size_idx_.average << std::endl;
      std::cout << " val: meas_size_idx_.segment " << meas_size_idx_.segment << std::endl;
      std::cout << " val: meas_size_idx_.phase " << meas_size_idx_.phase << std::endl;
      std::cout << " val: meas_size_idx_.set " << meas_size_idx_.set << std::endl;

      bool is_cartesian_sampling = (h.encoding[e].trajectory == ISMRMRD::TrajectoryType::CARTESIAN);
      bool is_epi_sampling= (h.encoding[e].trajectory == ISMRMRD::TrajectoryType::EPI);


      size_t E1, E2, SLC, AVE;
      E1=meas_size_idx_.kspace_encode_step_1;
      E2=meas_size_idx_.kspace_encode_step_2;
      SLC=meas_size_idx_.slice;
      AVE=meas_size_idx_.average;

      /*deja_vu.set_size(encoding,lNumberOfSlices_);
      deja_vu.zeros();

      deja_vu_epi_calib.set_size(1,lNumberOfSlices_);
      deja_vu_epi_calib.zeros();*/

      matrix_deja_vu_data_.create(E1, E2,SLC);
      matrix_deja_vu_epi_nav_.create(E1,E2,SLC);  // attention ne pas tenir compte de average , car average ==2 pour les lignes EPI

      matrix_deja_vu_data_.fill(0);
      matrix_deja_vu_epi_nav_.fill(0);

      return GADGET_OK;


  return GADGET_OK;
}




int WriteSingleBandFlagsGadget
::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
	  GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{


    e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    e2= m1->getObjectPtr()->idx.kspace_encode_step_2;
    slice= m1->getObjectPtr()->idx.slice;
    repetition= m1->getObjectPtr()->idx.repetition;
    set= m1->getObjectPtr()->idx.set;
    segment= m1->getObjectPtr()->idx.segment;
    phase= m1->getObjectPtr()->idx.phase;
    average= m1->getObjectPtr()->idx.average;
    user= m1->getObjectPtr()->idx.user[0];

       // std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

        // bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

          // std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

           if (repetition==0)
           {

               if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION))
               {
                   ///------------------------------------------------------------------------
                   /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
                   /// UK if acs calibration scan , do nothing
                   //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< repetition <<   " is_parallel yes " << std::endl;

               }
               else
               {

                   ///------------------------------------------------------------------------
                   /// FR
                   /// UK if not acs calibration , two cases, single band (SB) scans or multiband scans (MB)
                   /// As SB scans are always played before MB scans, an identical encoding value
                   /// This rule only works  without the presence of multiple contrasts , echos or average.
                   /// A proper solution could be to know the mdh tag for MB and SB scans



                   if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
                   {
                       matrix_deja_vu_epi_nav_(0, e2, slice)++;
                       //matrix_deja_vu_epi_nav_(0,e2,slice)=matrix_deja_vu_epi_nav_(0,e2,slice)+1;

                       //std::cout <<" e1 "  <<e1  <<" e2 "  <<e2  <<  " slice "  <<slice <<  " rep "<< repetition <<  " average "<< average <<  " is_corr yes " <<  " matrix "<<  matrix_deja_vu_epi_nav_(0, e2, slice)  << std::endl;

                       if (matrix_deja_vu_epi_nav_(0, e2, slice)>3)
                       {
                           // si c'est déjà vu c'est MB
                           m1->getObjectPtr()->idx.user[0]=0;
                       }
                       else
                       {
                           // si c'est pas déjà vu c'est SB
                           m1->getObjectPtr()->idx.user[0]=1;
                           // m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION); if commented -> array.data else -> array.ref

                       }
                   }
                   else
                   {
                       //matrix_deja_vu_data_(e1,e2,slice)=matrix_deja_vu_data_(e1,e2,slice)+1;

                       matrix_deja_vu_data_(e1, e2, slice)++;

                      // std::cout <<" e1 "  <<e1  <<" e2 "  <<e2  <<  " slice "  <<slice <<  " rep "<< repetition <<  " average "<< average  <<   " is_image yes " <<  " matrix "<<  matrix_deja_vu_data_(e1, e2, slice)  << std::endl;

                       if (matrix_deja_vu_data_(e1,e2, slice)>1)
                       {
                           // si c'est déjà vu c'est MB
                           m1->getObjectPtr()->idx.user[0]=0;
                       }
                       else
                       {
                           // si c'est pas déjà vu c'est SB
                           m1->getObjectPtr()->idx.user[0]=1;
                           // m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION); if commented -> array.data else -> array.ref
                       }
                   }
               }
           }
           else
           {
               ///------------------------------------------------------------------------
               /// FR  si on n'est pas à la repetition 1 c'est forcément des MB mais bon on va quand même être prudent
               /// UK  if repetition number if highter than 0 , it must be a MB scan

               // m1->getObjectPtr()->idx.user[0]=0; not necessary
           }


  //It is enough to put the first one, since they are linked
  if (this->next()->putq(m1) == -1) {
    m1->release();
    GERROR("AcquisitionPassthroughGadget::process, passing data on to next gadget");
    return -1;
  }

  return 0;
}
GADGET_FACTORY_DECLARE(WriteSingleBandFlagsGadget)
}



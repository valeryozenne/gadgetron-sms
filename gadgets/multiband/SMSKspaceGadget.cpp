/*
* SMSKspaceGadget.cpp
*
*  Created on: Dec 5, 2011
*      Author: hansenms
*/

#include "SMSKspaceGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"

namespace Gadgetron{

SMSKspaceGadget::SMSKspaceGadget() {
}

SMSKspaceGadget::~SMSKspaceGadget() {
}

int SMSKspaceGadget::process_config(ACE_Message_Block *mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;
    
    ISMRMRD::StudyInformation study_info;

    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;  /* Number of slices in one measurement */ //MANDATORY
    lNumberOfAverage_= e_limits.average? e_limits.average->maximum+1 : 1;

    nbChannels = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;

    std::cout <<  h.encoding[0].encodingLimits.kspace_encoding_step_1->maximum << std::endl;
    std::cout <<  h.encoding[0].encodingLimits.kspace_encoding_step_1->minimum << std::endl;

    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);
    GDEBUG("Matrix size after reconstruction: %d, %d, %d\n", dimensions_[0], dimensions_[1], dimensions_[2]);


    ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;

    acceFactorE1_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_1);
    acceFactorE2_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_2);
    acceFactorSlice_=2; //TODO à recuperer via les xml et xsl

    std::cout <<  " lNumberOfAverage_ " <<  lNumberOfAverage_  <<  " lNumberOfSlices_ " <<  lNumberOfSlices_   << std::endl;
    std::cout <<  " acceFactorE1_ " <<  acceFactorE1_  <<  " acceFactorE2_ " <<  acceFactorE2_ <<  " sliceFactor_ " << acceFactorSlice_  << std::endl;

    readout=dimensions_[0];
    encoding=dimensions_[1];

    deja_vu.set_size(encoding,lNumberOfSlices_);
    deja_vu.zeros();

    //parallel_calibration.set_size(readout*nbChannels,lNumberOfSlices_);
    slice_calibration.set_size(readout*nbChannels, encoding, lNumberOfSlices_);
    folded_image.set_size(readout*nbChannels, encoding, lNumberOfSlices_);

    str_home=GetHomeDirectory();
    //GDEBUG(" str_home: %s \n", str_home);

    ///------------------------------------------------------------------------
    /// FR
    /// UK
    find_encoding_dimension_for_SMS_calculation();

    ///------------------------------------------------------------------------
    /// FR
    /// UK

    flag_is_folded_slice.set_size(lNumberOfSlices_);
    flag_is_folded_slice.zeros();

    position_folded_slice.set_size(lNumberOfSlices_);
    position_folded_slice.zeros();

    lNumberOfGroup_=lNumberOfSlices_/acceFactorSlice_;

    std::cout <<  "lNumberOfGroup_"<< lNumberOfGroup_ << std::endl;

    position_debut_fin.set_size(acceFactorSlice_,lNumberOfGroup_);

    set_kernel_parameters();

    // initialization

    unfolded_image_with_blipped.set_size(readout*nbChannels,encoding,acceFactorSlice_);
    unfolded_image_with_blipped.zeros();

    unfolded_image_without_blipped.set_size(readout*nbChannels, encoding, acceFactorSlice_);
    unfolded_image_without_blipped_all.set_size(readout*nbChannels, encoding, lNumberOfSlices_);
    unfolded_image_without_blipped_inverse.set_size(readout*nbChannels, encoding, acceFactorSlice_);
    unfolded_image_without_blipped_inverse_all.set_size(readout*nbChannels, encoding, lNumberOfSlices_);

    recon_crop.set_size(acceFactorSlice_,1);
    recon_prep.set_size(acceFactorSlice_,1);
    recon_reshape.set_size(acceFactorSlice_,1);

    recon_crop_template.set_size(blocks_per_row, nbChannels, blocks_per_column);
    recon_prep_template.set_size(blocks_per_row+4, nbChannels, blocks_per_column+4);
    recon_reshape_template.set_size(readout*nbChannels,e1_size_for_sms_calculation,1);

    recon_crop_template.zeros();
    recon_prep_template.zeros();
    recon_reshape_template.zeros();

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        recon_crop(sms,0) = recon_crop_template;
        recon_prep(sms,0) = recon_prep_template;
        recon_reshape(sms,0) = recon_reshape_template;
    }


    if (lNumberOfSlices_==2 && acceFactorSlice_==2 )
    {
        slice_de_declenchement=1; //TODO peut être  position_folded_slice(end)

        flag_is_folded_slice(1)=1; //useless ?

        position_folded_slice(0)=1;

        position_debut_fin(0,0)=0;
        position_debut_fin(1,0)=1;

    }
    else if (lNumberOfSlices_==4 && acceFactorSlice_==2)
    {

        slice_de_declenchement=2; //TODO peut être  position_folded_slice(end)

        flag_is_folded_slice(0)=1; //useless ?
        flag_is_folded_slice(2)=1; //useless ?

        position_folded_slice(0)=0;
        position_folded_slice(1)=2;

        position_debut_fin(0,0)=0;
        position_debut_fin(1,0)=1;

        position_debut_fin(0,1)=2;
        position_debut_fin(1,1)=3;

    }

    else if (lNumberOfSlices_==24 && acceFactorSlice_==2)
    {

        slice_de_declenchement=lNumberOfGroup_; //TODO peut être  position_folded_slice(end)

        flag_is_folded_slice(0)=1; //useless ?
        flag_is_folded_slice(2)=1; //useless ?

        for (unsigned long grp = 0; grp < lNumberOfGroup_; grp++)
        {

        position_folded_slice(grp)=grp*2;

        position_debut_fin(0,grp)=grp*2;
        position_debut_fin(1,grp)=grp*2+1;

        }

    }

    std::cout <<  "position_debut_fin " << std::endl;

    std::cout <<  position_debut_fin  << std::endl;




    return GADGET_OK;
}


int SMSKspaceGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);
    //bool is_parallel_and imaging=m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);


    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice= m1->getObjectPtr()->idx.slice;
    unsigned int rep= m1->getObjectPtr()->idx.repetition;

    arma::cx_fvec tempo= as_arma_col(*m2->getObjectPtr());

    if (is_parallel)
    {
        ///------------------------------------------------------------------------
        /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
        /// UK
        //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }

    }
    else
    {
        deja_vu(e1,slice)++;

        if (deja_vu(e1,slice)>1)
        {
            //GDEBUG(" RecoSMS sms yes: %d  slice %d ,  rep %d  \n", e1, slice, rep );
            //std::cout <<   " is_last_scan_in_slice "<<  is_last_scan_in_slice  << " is_first_scan_in_slice "<<  is_first_scan_in_slice  <<  " is_last_in_encoding "<<  is_last_in_encoding  << std::endl;

            ///------------------------------------------------------------------------
            /// FR si on est ici nous sommes avec les lignes de kspace de l'image aliasée
            /// FR on stocke les lignes dans une matrice pour chaque repetition
            /// UK
            ///

            detect_flag(m1);

            folded_image.slice(slice).col(e1)=tempo;

            ///------------------------------------------------------------------------
            /// FR lorsqu'on est à la dernière ligne on peut commencer à déplier l'image
            /// UK

            //std::cout <<   " is_last_scan_in_slice "<<  is_last_scan_in_slice  << " slice "<<  slice  <<  " declenchement "<<  slice_de_declenchement  << std::endl;

            if (is_last_scan_in_slice && slice==slice_de_declenchement)
            {

                GDEBUG(" SMS Defolding is starting \n");

                //if (rep==0) {save_kspace_data(folded_image, "sms_");}


                ///------------------------------------------------------------------------
                /// FR normalement il faut faire un boucle sur le groupe de coupe MB
                /// UK

                for (unsigned long grp = 0; grp < lNumberOfGroup_; grp++)
                {
                    GDEBUG("Groupe: %d , position_folded_slice: %d   \n", grp ,  position_folded_slice(grp));

                    //std::cout << size(folded_image.slices(position_folded_slice(grp), position_folded_slice(grp))) << std::endl;

                    calculate_unfolded_image(folded_image.slices(position_folded_slice(grp), position_folded_slice(grp)));

                    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
                    {
                        GDEBUG("Copy SMS: %d  %d, position_slice: %d   \n", grp , sms, position_debut_fin(sms,grp));
                        unfolded_image_without_blipped_inverse_all.slice(position_debut_fin(sms,grp))=unfolded_image_without_blipped_inverse.slice(sms);
                    }

                    transfer_kspace_to_the_next_gadget(m1,m2, grp);
                }

            }
            else
            {

            }


            /*if( this->next()->putq(m1) < 0 ){
                GDEBUG("Failed to put message on queue\n");
                return GADGET_FAIL;
            }*/

        }
        else
        {
            //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " slice calibration " << is_parallel  << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

            if (rep==0)
            {
                ///------------------------------------------------------------------------
                /// FR stockage du kspace  [ readout*coil , encoding , nbSlicesTotal  ]
                /// UK

                slice_calibration.slice(slice).col(e1)=tempo;

                if (is_last_scan_in_slice && slice==(lNumberOfSlices_-1))
                {
                    GDEBUG("Kernel calculation is starting \n");

                    ///------------------------------------------------------------------------
                    /// FR normalement boucle sur le nombre de groupe, hypthese que les slices sont conjointes
                    /// UK

                    for (unsigned long grp = 0; grp < lNumberOfGroup_; grp++)
                    {
                        int debut=position_debut_fin(0,grp);
                        int fin=position_debut_fin(1,grp);

                        GDEBUG("Groupe: %d , debut: %d   , fin: %d  \n", grp ,  debut , fin);

                        calculate_kernel(slice_calibration.slices(debut,fin));

                        kernel_all_slices.slices(debut,fin)=kernel;

                    }

                    GDEBUG(" Kernel calculation is ending \n");

                }

            }
            else
            {
                std::cout << " on ne devrait pas venir ici, ce message indique qu'il y a un probleme"<< std::endl;
            }
        }
    }

    return 0;
}



int SMSKspaceGadget::transfer_kspace_to_the_next_gadget(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2 , int grp)
{

    //for (unsigned long sms = 0; sms < lNumberOfSlices_; sms++)

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {

        for (unsigned long i = 0; i < size(deja_vu,0); i++)
        {

            if(deja_vu(i,sms)>0)
            {

                GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* cm1 =
                        new GadgetContainerMessage<ISMRMRD::AcquisitionHeader>();

                *cm1->getObjectPtr() = *m1->getObjectPtr();

                GadgetContainerMessage<hoNDArray<std::complex<float> >  > *cm2 =
                        new GadgetContainerMessage<hoNDArray<std::complex<float> >  >();

                boost::shared_ptr< std::vector<size_t> > dims = m2->getObjectPtr()->get_dimensions();

                try{cm2->getObjectPtr()->create(dims.get());}
                catch (std::runtime_error &err){
                    GEXCEPTION(err,"Unable to create unsigned short storage in Extract Magnitude Gadget");
                    return GADGET_FAIL;
                }

                //arma::cx_fvec temp1= unfolded_image_without_blipped_inverse.slice(sms).col(i);
                //arma::cx_fvec temp1= slice_calibration.slice(sms).col(i);

                arma::cx_fvec temp1= unfolded_image_without_blipped_inverse_all.slice(position_debut_fin(sms,grp)).col(i);

                std::complex<float>* dst = cm2->getObjectPtr()->get_data_ptr();

                for (unsigned long j = 0; j < m2->getObjectPtr()->get_number_of_elements(); j++) {
                    dst[j]=temp1[j];
                }

                cm1->getObjectPtr()->idx.slice=position_debut_fin(sms,grp);
                cm1->getObjectPtr()->idx.kspace_encode_step_1=i;

                if (i==flag_encoding_first_scan_in_slice) //TODO
                {

                }
                else
                {
                    cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
                }

                if (i==flag_encoding_last_scan_in_slice)  //TODO
                {

                }
                else
                {
                    cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
                }

                cm1->cont(cm2);

                if( this->next()->putq(cm1) < 0 ){
                    GDEBUG("Failed to put message on queue\n");
                    return GADGET_FAIL;
                }
            }
        }
    }

    return GADGET_OK;
}


void SMSKspaceGadget::detect_flag(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1)
{

    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);
    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;

    if (is_last_in_encoding)
    {
        flag_encoding_last_in_encoding=e1;
        //std::cout<< "  flag_encoding_last_in_encoding " << flag_encoding_last_in_encoding  << std::endl;
    }

    if (is_first_scan_in_slice)
    {
        flag_encoding_first_scan_in_slice=e1;
        //std::cout<< "  flag_encoding_first_scan_in_slice " << flag_encoding_first_scan_in_slice  << std::endl;
    }

    if (is_last_scan_in_slice)
    {
        flag_encoding_last_scan_in_slice=e1;
        //std::cout<< "  flag_encoding_last_scan_in_slice " << flag_encoding_last_scan_in_slice  << std::endl;
    }

}


void SMSKspaceGadget::calculate_unfolded_image(arma::cx_fcube input)
{

    ///------------------------------------------------------------------------
    /// FR il faut commencer par enlever les données grappa en trop [readout*coil encoding/in_plane_acce slices]
    /// UK
    arma::cx_fcube folded_image_ready=remove_data_if_inplane_grappa(input);

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube [readout coil encoding]
    /// UK

    //TODO il faut rajouter une boucle sur les coupes aliasées
    arma::cx_fcube folded_image_reshape=preliminary_reshape(folded_image_ready.slice(0));

    ///------------------------------------------------------------------------
    /// FR on applique la fonction im2col
    /// UK
    arma::cx_fcube block_MB=im2col_version2(folded_image_reshape);

    block_MB.reshape( size(block_MB,0), size(block_MB,1)* size(block_MB,2),1 );

    block_MB.reshape( blocks_per_column, blocks_per_row, size(block_MB,1) );

    arma::cx_fcube lala(1,kernel_size*nbChannels,1);
    arma::cx_frowvec lili(kernel_size*nbChannels);


    for (unsigned long nidx1 = 0; nidx1 < blocks_per_column; nidx1++) //for nidx1=1:blocks_per_column
    {
        for (unsigned long nidx2 = 0; nidx2 < blocks_per_row; nidx2++) // for nidx2=1:blocks_per_row
        {
            for (unsigned int c = 0; c < nbChannels; c++)   //for nidy2=1:nc2   number of channels
            {
                //recon(nidx1,nidx2,1,nidy2,nidy3,nidy4)=permute(block_MB(nidx1,nidx2,:),[1 3 2]) * kernel(:,nidy2,nidy3,nidy4) ;

                lala=reshape(block_MB.tube(nidx1,nidx2),1,kernel_size*nbChannels,1);
                lili=lala.slice(0);

                for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
                {
                    recon_crop(sms,0).slice(nidx1).col(c).row(nidx2)=lili*kernel.slice(sms).col(c);
                }
            }
        }
    }



    int index_e1;

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        recon_prep(sms,0).subcube(2,0,2,blocks_per_row+1,nbChannels-1,blocks_per_column+1 )=recon_crop(sms,0); //TODO verifier le padding
        recon_reshape(sms,0)=reshape(recon_prep(sms,0),readout*nbChannels,e1_size_for_sms_calculation,1);

        for (unsigned long j = 0; j < e1_size_for_sms_calculation; j++) {

            index_e1=j*acceFactorE1_+1;
            //std::cout << " j " << j <<   "  index1 " << index1 << std::endl;
            unfolded_image_with_blipped.slice(sms).col(index_e1)=recon_reshape(sms,0).slice(0).col(j);

        }
    }

    unfolded_image_without_blipped=shifting_of_multislice_data(unfolded_image_with_blipped, -4);

    unfolded_image_without_blipped_inverse = unfolded_image_without_blipped;

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        unfolded_image_without_blipped_inverse.slice(sms)=unfolded_image_without_blipped.slice(acceFactorSlice_-sms-1);
    }



    /*for (unsigned long c = 0; c < nbChannels; c++)
    {
        //ceci est le numero de dynamique ie de 0 à N-1
        std::ostringstream indice_coil;
        indice_coil << c;
        str_c = indice_coil.str();
        Gadgetron::SaveVectorOntheDisk(vectorise(recon_slice0_ok.slice(c)), str_home, "Tempo/", "reco_sms_",  str_c, "0",  ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(recon_slice1_ok.slice(c)), str_home, "Tempo/", "reco_sms_",  str_c, "1",  ".bin");
    }*/
}


void SMSKspaceGadget::calculate_kernel(arma::cx_fcube input)
{

    ///------------------------------------------------------------------------
    /// FR nous sauvegardons le kspace sur le disque
    /// /// UK
    save_kspace_data(input, "scs_");

    ///------------------------------------------------------------------------
    /// FR il faut commencer par inverser les coupes, je ne sais pas pourquoi
    /// UK

    arma::cx_fcube slice_calibration_inverse=input;

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        slice_calibration_inverse.slice(sms)=slice_calibration.slice(acceFactorSlice_-sms-1);
    }

    ///------------------------------------------------------------------------
    /// FR blipped caipi shift
    /// UK

    // int center_k_space_sample=m1->getObjectPtr()->center_sample;
    // nous sauvegardons le kspace sur le disque
    //save_kspace_data(slice_calibration_inverse, "scs_inv_");

    arma::cx_fcube slice_calibration_inverse_blipped=shifting_of_multislice_data(slice_calibration_inverse, 4);

    // nous sauvegardons le kspace sur le disque
    // save_kspace_data(slice_calibration_inverse_blipped, "scs_inv_blip_");

    ///------------------------------------------------------------------------
    /// FR il faut commencer par enlever les données grappa en trop [readout*coil encoding/in_plane_acce slice]
    /// UK
    arma::cx_fcube slice_calibration_nohole=remove_data_if_inplane_grappa(slice_calibration_inverse_blipped);

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube de [readout*coil encoding slice] à  [encoding readout coil ] et on divise le resultat en field
    /// UK

    arma::field<arma::cx_fcube> block_SB(acceFactorSlice_,1);

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        block_SB(sms,0) = im2col(preliminary_reshape(slice_calibration_nohole.slice(sms)));
    }

    //arma::cx_fcube slice_calibration_reshape_slice_0=preliminary_reshape(slice_calibration_nohole.slice(0));
    //arma::cx_fcube slice_calibration_reshape_slice_1=preliminary_reshape(slice_calibration_nohole.slice(1));
    //Gadgetron::SaveVectorOntheDisk(vectorise(slice_calibration_reshape_slice_0.slice(0)),str_home, "Tempo/", "kernel_slice_calibration_reshape_",  "0", "0", ".bin");
    //Gadgetron::SaveVectorOntheDisk(vectorise(slice_calibration_reshape_slice_1.slice(0)),str_home, "Tempo/", "kernel_slice_calibration_reshape_",  "0", "1", ".bin");
    //std::cout <<  "size(slice_calibration_reshape) " << std::endl;
    //std::cout <<  size(slice_calibration_reshape_slice_0) << std::endl;
    //std::cout <<  size(slice_calibration_reshape_slice_1) << std::endl;
    //arma::cx_fcube block_SB_reshape_slice_0=im2col(slice_calibration_reshape_slice_0);
    //arma::cx_fcube block_SB_reshape_slice_1=im2col(slice_calibration_reshape_slice_1);
    //std::cout <<  "size(block_SB_reshape_slice_0 et 1)" << std::endl;
    //std::cout <<  size(block_SB_reshape_slice_0) << std::endl;
    //std::cout <<  size(block_SB_reshape_slice_1) << std::endl;

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(block_SB(0,0).slice(0)),str_home, "Tempo/", "kernel_block_SB_reshape_",  "0", "0", ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(block_SB(1,0).slice(0)),str_home, "Tempo/", "kernel_block_SB_reshape_",  "0", "1", ".bin");
    }

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube de [] à  []
    /// UK

    int milieu=round(float(kernel_size)/2)-1;
    //GDEBUG("Le milieu du kernel est %d  \n",milieu );

    arma::cx_fcube missing_data_slice(nb_pixels_per_image, nbChannels,acceFactorSlice_);

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        for (unsigned long p = 0; p < nb_pixels_per_image; p++)
        {
            for (unsigned long c = 0; c < nbChannels; c++)
            {
                missing_data_slice(p,c,sms)=block_SB(sms,0)(milieu,p,c);
                //missing_data_slice(p,c,0)=block_SB_reshape_slice_0(milieu,p,c);
                //missing_data_slice(p,c,1)=block_SB_reshape_slice_1(milieu,p,c);
            }
        }
    }

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(missing_data_slice.slice(0)),str_home, "Tempo/", "kernel_missing_data_slice_",  "0",  ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(missing_data_slice.slice(1)),str_home, "Tempo/", "kernel_missing_data_slice_",  "1",  ".bin");
    }

    arma::cx_fcube measured_data=block_SB(0,0);
    measured_data.zeros();

    for (unsigned long sms = 0; sms < acceFactorSlice_; sms++)
    {
        measured_data=measured_data+block_SB(sms,0);
    }

    if (debug)
    {
        for (unsigned long c = 0; c < nbChannels; c++)
        {
            //ceci est le numero de dynamique ie de 0 à N-1
            std::ostringstream indice_coil;
            indice_coil << c;
            str_c = indice_coil.str();
            Gadgetron::SaveVectorOntheDisk(vectorise(measured_data.slice(c)), str_home, "Tempo/", "kernel_measured_data_",  str_c,  ".bin");
        }
    }

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube [pixel , kernel , coil]
    /// UK

    arma::cx_fcube measured_data_ready(nb_pixels_per_image, kernel_size, nbChannels );

    for (unsigned long k = 0; k < kernel_size; k++)
    {
        for (unsigned long j = 0; j < nb_pixels_per_image; j++)
        {
            for (unsigned long c = 0; c < nbChannels; c++)
            {
                measured_data_ready(j,k,c)=measured_data(k,j,c);
            }
        }
    }

    ///------------------------------------------------------------------------
    /// FR son reformate dans un cube [pixel , kernel*coil , 1]
    /// UK

    measured_data_ready.reshape( size(measured_data_ready,0), size(measured_data_ready,1) * size(measured_data_ready,2) ,  1 );

    arma::cx_fmat measured_data_matrix=measured_data_ready.slice(0);

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(measured_data_matrix), str_home, "Tempo/", "kernel_measured_data_matrix_",  "0",  ".bin");
    }

    ///------------------------------------------------------------------------
    /// FR on prepare la résolution du systeme
    /// UK

    arma::cx_fmat A=(measured_data_matrix.t()*measured_data_matrix);

    ///------------------------------------------------------------------------
    /// FR résolution du systeme
    /// UK

    arma::cx_fmat CMK = pinv(A)*measured_data_matrix.t();

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(CMK),str_home, "Tempo/", "kernel_CMK_",  "0",  ".bin");
    }

    ///------------------------------------------------------------------------
    /// FR écriture du kernel
    /// UK

    //TODO rajouter les différentes implémentations
    for (unsigned long nidx1 = 0; nidx1 < size(CMK,0); nidx1++)
    {
        for (unsigned long c = 0; c < nbChannels; c++)
        {
            for (unsigned long s = 0; s < acceFactorSlice_; s++)
            {
                kernel.slice(s).col(c).row(nidx1)= CMK.row(nidx1)*missing_data_slice.slice(s).col(c) ;
            }
        }
    }

}



void SMSKspaceGadget::save_kspace_data(arma::cx_fcube input, std::string name)
{
    for (unsigned long s = 0; s < size(input,2); s++) {

        for (unsigned long e = 0; e < size(input,1); e++) {

            //ceci est le numero de dynamique ie de 0 à N-1
            std::ostringstream indice_encoding;
            indice_encoding << e;
            str_e = indice_encoding.str();

            //ceci est le numero de slice ie de 0 à S-1
            std::ostringstream indice_slice;
            indice_slice << s;
            str_s = indice_slice.str();

            Gadgetron::SaveVectorOntheDisk(input.slice(s).col(e),str_home, "Tempo/", name, str_e, str_s,  ".bin");
        }
    }

}


void SMSKspaceGadget::set_kernel_parameters(void)
{

    kernel_size=kernel_height.value()*kernel_width.value();

    kernel.set_size(kernel_size*nbChannels,nbChannels,acceFactorSlice_);
    kernel_all_slices.set_size(kernel_size*nbChannels,nbChannels,lNumberOfSlices_);

    blocks_per_row=readout-kernel_width.value()+1;

    blocks_per_column=e1_size_for_sms_calculation-(kernel_height.value())+1;

    nb_pixels_per_image=blocks_per_column*blocks_per_row;

    GDEBUG("Blocks_per_column:  %d,  blocks_per_row: %d, nb_pixels_per_image: %d\n", blocks_per_column, blocks_per_row,nb_pixels_per_image  );

}

void SMSKspaceGadget::find_encoding_dimension_for_SMS_calculation(void)
{
    if(acceFactorE1_>1)
    {
        e1_size_for_sms_calculation=round(encoding/acceFactorE1_);
    }
    else
    {
        e1_size_for_sms_calculation=encoding;
    }

    GDEBUG("Encoding size for sms calculation: %d\n", e1_size_for_sms_calculation);
}


arma::cx_fcube SMSKspaceGadget::preliminary_reshape(arma::cx_fmat input)
{
    ///------------------------------------------------------------------------
    /// FR
    /// UK

    arma::cx_fcube tempo(size(input,0),size(input,1),1 ) ;

    tempo.zeros();

    tempo.slice(0)=input;

    tempo.reshape( readout, nbChannels, size(input,1) );

    arma::cx_fcube output(e1_size_for_sms_calculation , readout, nbChannels );

    for (unsigned int r = 0; r < readout; r++)
    {
        for (unsigned int e1 = 0; e1 < e1_size_for_sms_calculation; e1++)
        {
            for (unsigned int c = 0; c < nbChannels; c++)
            {
                output(e1,r,c)= tempo(r,c,e1);
            }
        }
    }

    return output;

}

arma::cx_fcube SMSKspaceGadget::remove_data_if_inplane_grappa(arma::cx_fcube input)
{

    ///------------------------------------------------------------------------
    /// FR
    /// UK

    GDEBUG("Encoding size for sms calculation: %d\n", e1_size_for_sms_calculation);

    arma::cx_fcube  output(size(input,0),e1_size_for_sms_calculation,size(input,2) );

    if(acceFactorE1_>1)
    {
        GDEBUG("Changing encoding size for sms calculation from %d to %d\n",encoding, e1_size_for_sms_calculation);

        int indice;

        for (unsigned int s = 0; s < size(input,2); s++)
        {
            // std::cout << s << std::endl;
            // ici on peut ajouter une tableau qui dit si on continue ou pas

            for (unsigned int j = 1; j < size(input,1); j+=acceFactorE1_)
            {
                indice = ((j+1)/acceFactorE1_)-1;
                output.slice(s).col(indice)=input.slice(s).col(j);
            }
        }
    }
    else
    {
        GDEBUG("Keeping encoding size for sms calculation to %d\n", e1_size_for_sms_calculation);
        output=input;
    }

    return output;

}


arma::cx_fcube SMSKspaceGadget::im2col_version2(arma::cx_fcube input)
{
    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK

    //std::cout <<   size(input) << std::endl;

    GDEBUG("Blocks_per_column:  %d,  blocks_per_row: %d, nb_pixels_per_image: %d\n", blocks_per_column, blocks_per_row,nb_pixels_per_image  );

    int xB=blocks_per_column;
    int yB=blocks_per_row;

    arma::cx_fcube block_MB(xB*yB,kernel_size,nbChannels );
    block_MB.zeros();

    int rowIdx, colIdx;

    for (unsigned int c = 0; c < nbChannels; c++)
    {

        for (unsigned int j = 0; j < yB; j++)
        {
            for (unsigned int i = 0; i < xB; i++)
            {
                rowIdx = i + j*xB;

                for (unsigned int yy = 0; yy < kernel_height.value(); yy++)
                {
                    for (unsigned int xx = 0; xx < kernel_width.value(); xx++)
                    {
                        colIdx = xx + yy*kernel_width.value();

                        //if (c==0 && i==0 && j==0)     std::cout <<   xx <<  " " <<  yy<< " "<<  colIdx <<std::endl;

                        block_MB(rowIdx, colIdx,c)=input(i+xx, j+yy,c);
                    }
                }
            }
        }
    }

    return block_MB;
}


arma::cx_fcube SMSKspaceGadget::shifting_of_multislice_data(arma::cx_fcube input, int PE_shift)
{

    int center_k_space_sample=round(float(encoding)/2);

    arma::fvec index_imag = arma::linspace<arma::fvec>( 1, encoding, encoding )  - center_k_space_sample ;

    arma::cx_fvec phase(size(slice_calibration,1));
    phase.zeros();
    phase.set_imag(index_imag);

    arma::cx_fcube output=input;
    output.zeros();

    double caipi_factor;

    for (unsigned int s = 0; s < acceFactorSlice_; s++)
    {
        caipi_factor=2*arma::datum::pi/PE_shift*s; // if s==0 rien, if s==1 2*pi/shift

        GDEBUG("Blipped caipi shift: slice: %d , caipi_factor: %f \n", s, caipi_factor);

        arma::cx_fvec lala=exp(phase*caipi_factor);

        for (unsigned int r = 0; r < size(input,0); r++)
        {
            output.slice(s).row(r)=input.slice(s).row(r)%lala.t();
        }
    }

    return output;
}

arma::cx_fcube SMSKspaceGadget::im2col(arma::cx_fcube input)
{
    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK

    int xB=blocks_per_column;
    int yB=blocks_per_row;

    arma::cx_fcube block_MB(kernel_size,xB*yB,nbChannels );
    block_MB.zeros();

    int rowIdx, colIdx;

    for (unsigned int c = 0; c < nbChannels; c++)
    {
        for (unsigned int j = 0; j < yB; j++)
        {
            for (unsigned int i = 0; i < xB; i++)
            {
                rowIdx = i + j*xB;

                for (unsigned int yy = 0; yy < kernel_height.value(); yy++)
                {
                    for (unsigned int xx = 0; xx < kernel_width.value(); xx++)
                    {
                        colIdx = xx + yy*kernel_width.value();

                        //if (c==0 && i==0 && j==0)     std::cout <<   xx <<  " " <<  yy<< " "<<  colIdx <<std::endl;

                        block_MB(colIdx, rowIdx,c)=input(i+xx, j+yy,c);
                    }
                }
            }
        }
    }

    return block_MB;
}




GADGET_FACTORY_DECLARE(SMSKspaceGadget)
}

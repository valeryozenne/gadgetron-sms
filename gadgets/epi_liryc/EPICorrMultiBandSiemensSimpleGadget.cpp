//    TO DO: - For 3D sequences (E2>1), the 3 navigators should be equivalent for all e2 partition
//             encoding steps (other than the mean phase).  So we should average across them.  I guess
//             one way to do so would be to include all partition encoding steps (that have been acquired
//             up to that one), also from the previous repetitions, in the robust fit, being careful with
//             the column representing the slope.  The problem with that is that the matrix to invert is
//             longer, so it could take longer to compute.
//           - Test the case that more repetitions are sent than the number specified in the xml header.

#include "EPICorrMultiBandSiemensSimpleGadget.h"
#include "ismrmrd/xml.h"
#include "hoNDArray_fileio.h"
#include "mri_core_utility_interventional.h"
#include "mri_core_multiband.h"

namespace Gadgetron{

#define OE_PHASE_CORR_POLY_ORDER 4

EPICorrMultiBandSiemensSimpleGadget::EPICorrMultiBandSiemensSimpleGadget() {}
EPICorrMultiBandSiemensSimpleGadget::~EPICorrMultiBandSiemensSimpleGadget() {}

int EPICorrMultiBandSiemensSimpleGadget::process_config(ACE_Message_Block* mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    if (h.encoding.size() == 0) {
        GDEBUG("Number of encoding spaces: %d\n", h.encoding.size());
        GDEBUG("This Gadget needs an encoding description\n");
        return GADGET_FAIL;
    }



    Gadgetron::DeleteTemporaryFiles( str_home, "Tempo/", "corr*" );

    // Get the encoding space and trajectory description
    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;


    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);

    readout=dimensions_[0];
    encoding=dimensions_[1];



    ///////////////////////////////

    if (h.encoding[0].trajectoryDescription) {
        traj_desc = *h.encoding[0].trajectoryDescription;
    } else {
        GDEBUG("Trajectory description missing");
        return GADGET_FAIL;
    }

    if (traj_desc.identifier != "ConventionalEPI") {
        GDEBUG("Expected trajectory description identifier 'ConventionalEPI', not found.");
        return GADGET_FAIL;
    }


    for (std::vector<ISMRMRD::UserParameterLong>::iterator i (traj_desc.userParameterLong.begin()); i != traj_desc.userParameterLong.end(); ++i) {
        if (i->name == "numberOfNavigators") {
            numNavigators_ = i->value;
        } else if (i->name == "etl") {
            etl_ = i->value;
        }
    }

    // Make sure the reference navigator is properly set:
    if (referenceNavigatorNumber.value() > (numNavigators_-1)) {
        GDEBUG("Reference navigator number is larger than number of navigators acquired.");
        return GADGET_FAIL;
    }

    // Initialize arrays needed for temporal filtering, if requested:
    GDEBUG_STREAM("navigatorParameterFilterLength = " << navigatorParameterFilterLength.value());
    if (navigatorParameterFilterLength.value() > 1)
    {
        init_arrays_for_nav_parameter_filtering( e_limits );
    }

    verboseMode_ = verboseMode.value();

    corrComputed_ = false;
    navNumber_ = -1;
    epiEchoNumber_ = -1;

    ///////////////////////////////

    str_home=GetHomeDirectory();
    deal_with_inline_or_offline_situation(h);

    if (OEPhaseCorrectionMode.value().compare("mean")==0    )
    {
        GDEBUG(" OEPhaseCorrectionMode : mean \n");
    }

    if (OEPhaseCorrectionMode.value().compare("linear")==0    )
    {
        GDEBUG(" OEPhaseCorrectionMode : linear \n");
    }

    if (OEPhaseCorrectionMode.value().compare("polynomial")==0    )
    {
        GDEBUG(" OEPhaseCorrectionMode : polynomial \n");
    }


    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;  /* Number of slices in one measurement */ //MANDATORY
    lNumberOfAverage_= e_limits.average? e_limits.average->maximum+1 : 1;
    lNumberOfStacks_= lNumberOfSlices_/MB_factor_;
    lNumberOfChannels_ = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;

    bool no_reordering=0;

    order_of_acquisition_mb = Gadgetron::map_interleaved_acquisitions_mb(lNumberOfStacks_, no_reordering );

    order_of_acquisition_sb = Gadgetron::map_interleaved_acquisitions(lNumberOfSlices_, no_reordering );

    indice_mb = sort_index( order_of_acquisition_mb );
    //std::cout << " EPI " <<  indice_mb << std::endl;

    indice_sb = sort_index( order_of_acquisition_sb );
    //std::cout <<  " EPI " <<  indice_sb << std::endl;

    find_encoding_dimension_for_epi_calculation();

    // Reordering
    // Assuming interleaved SMS acquisition, define the acquired slices for each multislice band
    MapSliceSMS=Gadgetron::get_map_slice_single_band( MB_factor_,  lNumberOfStacks_,  order_of_acquisition_mb,  no_reordering);

    vec_MapSliceSMS=vectorise(MapSliceSMS);

    ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;

    acceFactorE1_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_1);
    acceFactorE2_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_2);

    corrpos_mean_.set_size( readout , lNumberOfStacks_);
    corrneg_mean_.set_size( readout , lNumberOfStacks_);

    corrpos_all_.set_size( readout , lNumberOfSlices_);
    corrneg_all_.set_size( readout , lNumberOfSlices_);

    corrpos_no_exp_mean_.set_size( readout , lNumberOfStacks_);
    corrneg_no_exp_mean_.set_size( readout , lNumberOfStacks_);

    corrpos_no_exp_save_.set_size( readout , lNumberOfSlices_);
    corrneg_no_exp_save_.set_size( readout , lNumberOfSlices_);

    corrpos_mean_.zeros();
    corrneg_mean_.zeros();

    corrpos_all_.zeros();
    corrneg_all_.zeros();

    corrpos_no_exp_mean_.zeros();
    corrneg_no_exp_mean_.zeros();

    corrpos_no_exp_save_.zeros();
    corrneg_no_exp_save_.zeros();

    deja_vu_stacks.set_size(lNumberOfStacks_);
    deja_vu_stacks.zeros();

    slice_calibration.set_size(readout*lNumberOfChannels_, encoding, lNumberOfSlices_);
    slice_calibration.zeros();

    matrix_encoding.set_size(encoding,lNumberOfSlices_);
    matrix_encoding.zeros();

    reverse_encoding.set_size(encoding,lNumberOfSlices_);
    reverse_encoding.zeros();

    compteur_sb_sum=0;
    compteur_mb_sum=0;

    flag_average_ghost_niquist_correction_are_available=0;

    ///////////////////////////////
    /// \brief GDEBUG_STREAM
    ///
    ///
    GDEBUG_STREAM("EPICorrMultiBandSiemensSimpleGadget configured");
    return 0;
}

int EPICorrMultiBandSiemensSimpleGadget::process(
        GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
        GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{

    //GDEBUG_STREAM("Nav: " << navNumber_ << "    " << "Echo: " << epiEchoNumber_ << std::endl);

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice= m1->getObjectPtr()->idx.slice;
    unsigned int rep= m1->getObjectPtr()->idx.repetition;

    bool is_acq_single_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
    bool is_acq_multi_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);

    // Get a reference to the acquisition header
    ISMRMRD::AcquisitionHeader &hdr = *m1->getObjectPtr();

    // Pass on the non-EPI data (e.g. FLASH Calibration)
    if (hdr.encoding_space_ref > 0) {
        // It is enough to put the first one, since they are linked
        if (this->next()->putq(m1) == -1) {
            m1->release();
            GERROR("EPICorrMultiBandSiemensSimpleGadget::process, passing data on to next gadget");
            return -1;
        }
        return 0;
    }

    // We have data from encoding space 0.

    // Make an armadillo matrix of the data
    arma::cx_fmat adata = as_arma_matrix(*m2->getObjectPtr());

    // Check to see if the data is a navigator line or an imaging line
    if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA)) {

        // Increment the navigator counter
        navNumber_ += 1;

        // If the number of navigators per shot is exceeded, then
        // we are at the beginning of the next shot
        if (navNumber_ == numNavigators_) {
            corrComputed_ = false;
            navNumber_ = 0;
            epiEchoNumber_ = -1;
        }

        int Nx_ = adata.n_rows;

        // If we are at the beginning of a shot, then initialize
        if (navNumber_==0) {
            // Set the size of the corrections and storage arrays
            corrB0_.set_size(  Nx_ );
            corrpos_.set_size( Nx_ );
            corrneg_.set_size( Nx_ );

            navdata_.set_size( Nx_, hdr.active_channels, numNavigators_);
            navdata_.zeros();
            // Store the first navigator's polarity
            startNegative_ = hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE);
        }

        // Store the navigator data
        navdata_.slice(navNumber_) = adata;



        // If this is the last of the navigators for this shot, then
        // compute the correction operator
        if (navNumber_ == (numNavigators_-1)) {
            arma::cx_fvec ctemp =  arma::zeros<arma::cx_fvec>(Nx_);    // temp column complex
            arma::fvec tvec = arma::zeros<arma::fvec>(Nx_);            // temp column real
            arma::fvec x = arma::linspace<arma::fvec>(-0.5, 0.5, Nx_); // Evenly spaced x-space locations
            arma::fmat X;
            if ( OEPhaseCorrectionMode.value().compare("polynomial")==0 )
            {
                X  = arma::zeros<arma::fmat>( Nx_ ,OE_PHASE_CORR_POLY_ORDER+1);
                X.col(0) = arma::ones<arma::fvec>( Nx_ );
                X.col(1) = x;                       // x
                X.col(2) = arma::square(x);         // x^2
                X.col(3) = x % X.col(2);            // x^3
                X.col(4) = arma::square(X.col(2));  // x^4
            }
            int p; // counter

            // mean of the reference navigator (across RO and channels):
            std::complex<float> navMean = arma::mean( arma::vectorise( navdata_.slice(referenceNavigatorNumber.value()) ) );
            //GDEBUG_STREAM("navMean = " << navMean);

            // for clarity, we'll use the following when filtering navigator parameters:
            size_t set, slc, exc;
            if (navigatorParameterFilterLength.value() > 1)
            {
                set = hdr.idx.set;
                slc = hdr.idx.slice;
                // Careful: kspace_encode_step_2 for a navigator is always 0, and at this point we
                //          don't have access to the kspace_encode_step_2 for the next line.  Instead,
                //          keep track of the excitation number for this set and slice:
                //size_t e2  = hdr.idx.kspace_encode_step_2;
                //size_t rep = hdr.idx.repetition;
                exc = excitNo_[slc][set];   // excitation number with this same specific set and slc
                //GDEBUG_STREAM("Excitation number:" << exc << "; slice: " << slc);

                // If, for whatever reason, we are getting more repetitions than the header
                //   specified, increase the size of the array to accomodate:
                if ( exc >= (Nav_mag_.get_size(0)/E2_) )
                {
                    this->increase_no_repetitions( 100 );     // add 100 volumes more, to be safe
                }
                Nav_mag_(exc, set, slc) = std::abs(navMean);
            }


            /////////////////////////////////////
            //////      B0 correction      //////
            /////////////////////////////////////

            if ( B0CorrectionMode.value().compare("none")!=0 )    // If B0 correction is requested
            {
                // Accumulate over navigator pairs and sum over coils
                // this is the average phase difference between consecutive odd or even navigators
                for (p=0; p<numNavigators_-2; p++)
                {
                    ctemp += arma::sum(arma::conj(navdata_.slice(p)) % navdata_.slice(p+2),1);
                }

                // Perform the fit:
                float slope = 0.;
                float intercept = 0.;
                if ( (B0CorrectionMode.value().compare("mean")==0)  ||
                     (B0CorrectionMode.value().compare("linear")==0) )
                {
                    // If a linear term is requested, compute it first (in the complex domain):
                    if (B0CorrectionMode.value().compare("linear")==0)
                    {          // Robust fit to a straight line:
                        slope = (Nx_-1) * std::arg(arma::cdot(ctemp.rows(0,Nx_-2), ctemp.rows(1,Nx_-1)));
                        //GDEBUG_STREAM("Slope = " << slope << std::endl);
                        // If we need to filter the estimate:
                        if (navigatorParameterFilterLength.value() > 1)
                        {
                            // (Because to estimate the intercept (constant term) we need to use the slope estimate,
                            //   we want to filter it first):
                            //   - Store the value in the corresponding array (we want to store it before filtering)
                            B0_slope_(exc, set, slc) = slope;
                            //   - Filter parameter:
                            slope = filter_nav_correction_parameter( B0_slope_, Nav_mag_, exc, set, slc, navigatorParameterFilterLength.value() );
                        }

                        // Correct for the slope, to be able to compute the average phase:
                        ctemp = ctemp % arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>( Nx_ ), -slope*x));
                    }   // end of the B0CorrectionMode == "linear"

                    // Now, compute the mean phase:
                    intercept = std::arg(arma::sum(ctemp));
                    //GDEBUG_STREAM("Intercept = " << intercept << std::endl);
                    if (navigatorParameterFilterLength.value() > 1)
                    {
                        //   - Store the value found in the corresponding array:
                        B0_intercept_(exc, set, slc) = intercept;
                        //   - Filter parameters:
                        // Filter in the complex domain (last arg:"true"), to avoid smoothing across phase wraps:
                        intercept = filter_nav_correction_parameter( B0_intercept_, Nav_mag_, exc, set, slc, navigatorParameterFilterLength.value(), true );
                    }

                    // Then, our estimate of the phase:
                    tvec = slope*x + intercept;

                }       // end of B0CorrectionMode == "mean" or "linear"

                // The B0 Correction:
                // 0.5* because what we have calculated was the phase difference between every other navigator
                corrB0_ = arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>(ctemp.n_rows), -0.5*tvec));

            }        // end of B0CorrectionMode != "none"
            else
            {      // No B0 correction:
                corrB0_.ones();
            }


            ////////////////////////////////////////////////////
            //////      Odd-Even correction -- Phase      //////
            ////////////////////////////////////////////////////

            if (OEPhaseCorrectionMode.value().compare("none")!=0)    // If Odd-Even phase correction is requested
            {
                // Accumulate over navigator triplets and sum over coils
                // this is the average phase difference between odd and even navigators
                // Note: we have to correct for the B0 evolution between navigators before
                ctemp.zeros();      // set all elements to zero
                for (p=0; p<numNavigators_-2; p=p+2)
                {
                    ctemp += arma::sum( arma::conj( navdata_.slice(p)/repmat(corrB0_,1,navdata_.n_cols) + navdata_.slice(p+2)%repmat(corrB0_,1,navdata_.n_cols) ) % navdata_.slice(p+1),1);
                }

                float slope = 0.;
                float intercept = 0.;
                if ( (OEPhaseCorrectionMode.value().compare("mean")==0      ) ||
                     (OEPhaseCorrectionMode.value().compare("linear")==0    ) ||
                     (OEPhaseCorrectionMode.value().compare("polynomial")==0) )
                {
                    // If a linear term is requested, compute it first (in the complex domain):
                    // (This is important in case there are -pi/+pi phase wraps, since a polynomial
                    //  fit to the phase will not work)
                    if ( (OEPhaseCorrectionMode.value().compare("linear")==0    ) ||
                         (OEPhaseCorrectionMode.value().compare("polynomial")==0) )
                    {          // Robust fit to a straight line:
                        slope = (Nx_-1) * std::arg(arma::cdot(ctemp.rows(0,Nx_-2), ctemp.rows(1,Nx_-1)));

                        // If we need to filter the estimate:
                        if (navigatorParameterFilterLength.value() > 1)
                        {
                            // (Because to estimate the intercept (constant term) we need to use the slope estimate,
                            //   we want to filter it first):
                            //   - Store the value in the corresponding array (we want to store it before filtering)
                            OE_phi_slope_(exc, set, slc) = slope;
                            //   - Filter parameter:
                            slope = filter_nav_correction_parameter( OE_phi_slope_, Nav_mag_, exc, set, slc, navigatorParameterFilterLength.value() );
                        }

                        // Now correct for the slope, to be able to compute the average phase:
                        ctemp = ctemp % arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>( Nx_ ), -slope*x));
                        // at this point we should have got rid of any -pi/+pi phase wraps.
                    }   // end of the OEPhaseCorrectionMode == "linear" or "polynomial"

                    // Now, compute the mean phase:
                    intercept = std::arg(arma::sum(ctemp));
                    //GDEBUG_STREAM("Intercept = " << intercept << std::endl);
                    if (navigatorParameterFilterLength.value() > 1)
                    {
                        //   - Store the value found in the corresponding array:
                        OE_phi_intercept_(exc, set, slc) = intercept;
                        //   - Filter parameters:
                        // Filter in the complex domain ("true"), to avoid smoothing across phase wraps:
                        intercept = filter_nav_correction_parameter( OE_phi_intercept_, Nav_mag_, exc, set, slc, navigatorParameterFilterLength.value(), true );
                    }

                    // Then, our estimate of the phase:
                    tvec = slope*x + intercept;

                    // If a polynomial fit is requested:
                    if (OEPhaseCorrectionMode.value().compare("polynomial")==0)
                    {
                        // Fit the residuals (i.e., after removing the linear trend) to a polynomial.
                        // You cannot fit the phase directly to the polynomial because it doesn't work
                        //   in cases that the phase wraps across the image.
                        // Since we have already removed the slope (in the if OEPhaseCorrectionMode
                        //   == "linear" or "polynomial" step), just remove the constant phase:
                        ctemp = ctemp % arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>( Nx_ ), -intercept*arma::ones<arma::fvec>( Nx_ )));

                        // Use the magnitude of the average odd navigator as weights:
                        arma::fvec ctemp_odd  = arma::zeros<arma::fvec>(Nx_);    // temp column complex for odd  magnitudes
                        for (int p=0; p<numNavigators_-2; p=p+2)
                        {
                            ctemp_odd  += ( arma::sqrt(arma::sum(arma::square(arma::abs(navdata_.slice(p))),1)) + arma::sqrt(arma::sum(arma::square(arma::abs(navdata_.slice(p+2))),1)) )/2;
                        }

                        arma::fmat WX     = arma::diagmat(ctemp_odd) * X;   // Weighted polynomial matrix
                        arma::fvec Wctemp( Nx_ );                           // Weighted phase residual
                        for (int p=0; p<Nx_; p++)
                        {
                            Wctemp(p) = ctemp_odd(p) * std::arg(ctemp(p));
                        }

                        // Solve for the polynomial coefficients:
                        arma::fvec phase_poly_coef = arma::solve( WX , Wctemp );

                        if (navigatorParameterFilterLength.value() > 1)
                        {
                            for (size_t i = 0; i < OE_phi_poly_coef_.size(); ++i)
                            {
                                //   - Store the value found in the corresponding array:
                                OE_phi_poly_coef_[i](exc, set, slc) = phase_poly_coef(i);

                                //   - Filter parameters:
                                phase_poly_coef(i) = filter_nav_correction_parameter( OE_phi_poly_coef_[i], Nav_mag_, exc, set, slc, navigatorParameterFilterLength.value() );
                            }
                            //GDEBUG_STREAM("OE_phi_poly_coef size: " << OE_phi_poly_coef_.size());
                        }

                        // Then, update our estimate of the phase correction:
                        tvec += X * phase_poly_coef;     // ( Note the "+=" )

                    }   // end of OEPhaseCorrectionMode == "polynomial"

                }       // end of OEPhaseCorrectionMode == "mean", "linear" or "polynomial"

                if (!startNegative_) {
                    // if the first navigator is a positive readout, we need to flip the sign of our correction
                    tvec = -1.0*tvec;
                }
            }    // end of OEPhaseCorrectionMode != "none"
            else
            {      // No OEPhase correction:
                tvec.zeros();
            }

            // Odd and even phase corrections
            corrpos_ = arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>(Nx_), -0.5*tvec));
            corrneg_ = arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>(Nx_), +0.5*tvec));

            if (is_acq_single_band) //sinon c'est inutile
            {
                //compteur qui permet de savoir si on a bien attend la dernier sb
                //TODO cette condition doit être respecté
                compteur_sb_sum++;

                //TODO conditions sur compteur_sb_sum à respecter
                fonction_qui_sauvegarde_en_memoire_et_somme_les_corrections(slice, Nx_, tvec);

            }

            corrComputed_ = true;

            // Increase the excitation number for this slice and set (to be used for the next shot)
            if (navigatorParameterFilterLength.value() > 1) {
                excitNo_[slc][set]++;
            }
        }

    }
    else {
        // Increment the echo number
        epiEchoNumber_ += 1;

        if (epiEchoNumber_ == 0)
        {
            // For now, we will correct the phase evolution of each EPI line, with respect
            //   to the first line in the EPI readout train (echo 0), due to B0 inhomogeneities.
            //   That is, the reconstructed images will have the phase that the object had at
            //   the beginning of the EPI readout train (excluding the phase due to encoding),
            //   multiplied by the coil phase.
            // Later, we could add the time between the excitation and echo 0, or between one
            //   of the navigators and echo 0, to correct for phase differences from shot to shot.
            //   This will be important for multi-shot EPI acquisitions.
            RefNav_to_Echo0_time_ES_ = 0;
        }

        // nous allons imprimer la correction EPI moyenne pour toutes les coupes
        // TODO verifier les conditions
        if (is_acq_single_band && is_first_scan_in_slice)
        {
            // on imprime au fur et à mesure pour chaque coupe
            fonction_qui_sauvegarde_sur_le_disk_les_corrections_par_coupes(slice );

            // et on imprime les valeurs moyennes par stacks.
            // pour cela il faut avoir calculé la moyenne

            //fonction_qui_sauvegarde_sur_le_disk_les_corrections_par_stacks(slice );

        }

        if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION))
        {
            // on fait la correction normalement

            // Apply the correction
            // We use the armadillo notation that loops over all the columns
            if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)) {
                // Negative readout
                for (int p=0; p<adata.n_cols; p++) {
                    //adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrneg_);
                    adata.col(p) %=  corrneg_;
                }
                // Now that we have corrected we set the readout direction to positive
                hdr.clearFlag(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE);
            }
            else {
                // Positive readout
                for (int p=0; p<adata.n_cols; p++) {
                    //adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrpos_);
                    adata.col(p) %=  corrpos_;
                }
            }

            // Pass on the imaging data
            // TODO: this should be controlled by a flag
            if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA)) {
                m1->release();
            }
            else {
                // It is enough to put the first one, since they are linked
                if (this->next()->putq(m1) == -1) {
                    m1->release();
                    GERROR("EPICorrMultiBandSiemensSimpleGadget::process, passing data on to next gadget");
                    return -1;
                }
            }

        }
        else if (is_acq_single_band)
        {

            ///////////////////////////////////////
            /// mode test
            ///////////////////////////////////////
            // on bufferise toutes les coupes
            // et on les renvoie à la dernière dynamique

            // on applique pas de correction , qui sera effectuee dans le fichier matlab

            if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA)) {
                m1->release();
            }
            else
            {
                // It is enough to put the first one, since they are linked
                if (this->next()->putq(m1) == -1) {
                    m1->release();
                    GERROR("EPICorrMultiBandSiemensSimpleGadget::process, passing data on to next gadget");
                    return -1;
                }
            }


        }
        else if (is_acq_multi_band)
        {

            // on applique pas de correction , qui sera effectuee dans le fichier matlab

            if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA)) {
                m1->release();
            }
            else {
                // It is enough to put the first one, since they are linked
                if (this->next()->putq(m1) == -1) {
                    m1->release();
                    GERROR("EPICorrMultiBandSiemensSimpleGadget::process, passing data on to next gadget");
                    return -1;
                }
            }
        }

        // Pass on the imaging data
        // TODO: this should be controlled by a flag
        if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA)) {
            m1->release();
        }
        /*else {
            // It is enough to put the first one, since they are linked
            if (this->next()->putq(m1) == -1) {
                m1->release();
                GERROR("EPICorrMultiBandSiemensSimpleGadget::process, passing data on to next gadget");
                return -1;
            }
        }*/

    }


    return 0;
}


//////////////////////////////////////////////////////////
//
// init_arrays_for_nav_parameter_filtering
//
//    function to initialize the arrays that will be used for the navigator parameters filtering
//    - e_limits: encoding limits

void EPICorrMultiBandSiemensSimpleGadget::init_arrays_for_nav_parameter_filtering( ISMRMRD::EncodingLimits e_limits )
{
    // TO DO: Take into account the acceleration along E2:

    E2_  = e_limits.kspace_encoding_step_2 ? e_limits.kspace_encoding_step_2->maximum - e_limits.kspace_encoding_step_2->minimum + 1 : 1;
    size_t REP = e_limits.repetition ? e_limits.repetition->maximum - e_limits.repetition->minimum + 1 : 1;
    size_t SET = e_limits.set        ? e_limits.set->maximum        - e_limits.set->minimum        + 1 : 1;
    size_t SLC = e_limits.slice      ? e_limits.slice->maximum      - e_limits.slice->minimum      + 1 : 1;
    // NOTE: For EPI sequences, "segment" indicates odd/even readout, so we don't need a separate dimension for it.
    GDEBUG_STREAM("E2: " << E2_ << "; SLC: " << SLC << "; REP: " << REP << "; SET: " << SET);

    // For 3D sequences, the e2 index in the navigator is always 0 (there is no phase encoding in
    //   the navigator), so we keep track of the excitation number for each slice and set) to do
    //   the filtering>
    excitNo_.resize(SLC);
    for (size_t i = 0; i < SLC; ++i)
    {
        excitNo_[i].resize( SET, size_t(0) );
    }

    // For 3D sequences, all e2 phase encoding steps excite the whole volume, so the
    //   navigators should be the same.  So when we filter across repetitions, we have
    //   to do it also through e2.  Bottom line: e2 and repetition are equivalent.
    Nav_mag_.create(         E2_*REP, SET, SLC);
    B0_intercept_.create(    E2_*REP, SET, SLC);
    if (B0CorrectionMode.value().compare("linear")==0)
    {
        B0_slope_.create(        E2_*REP, SET, SLC);
    }
    OE_phi_intercept_.create(E2_*REP, SET, SLC);
    if ((OEPhaseCorrectionMode.value().compare("linear")==0    ) ||
            (OEPhaseCorrectionMode.value().compare("polynomial")==0) )
    {
        OE_phi_slope_.create(    E2_*REP, SET, SLC);
        if (OEPhaseCorrectionMode.value().compare("polynomial")==0)
        {
            OE_phi_poly_coef_.resize( OE_PHASE_CORR_POLY_ORDER+1 );
            for (size_t i = 0; i < OE_phi_poly_coef_.size(); ++i)
            {
                OE_phi_poly_coef_[i].create( E2_*REP, SET, SLC);
            }
        }
    }

    // Armadillo vector of evenly-spaced timepoints to filter navigator parameters:
    t_ = arma::linspace<arma::fvec>( 0, navigatorParameterFilterLength.value()-1, navigatorParameterFilterLength.value() );

}


////////////////////////////////////////////////////
//
//  filter_nav_correction_parameter
//
//    funtion to filter (over e2/repetition number) a navigator parameter.
//    - nav_corr_param_array: array of navigator parameters
//    - weights_array       : array with weights for the filtering
//    - exc                 : current excitation number (for this set and slice)
//    - set                 : set of the array to filter (current one)
//    - slc                 : slice of the array to filter (current one)
//    - Nt                  : number of e2/timepoints/repetitions to filter
//    - filter_in_complex_domain : whether to filter in the complex domain, to avoid +/- pi wraps (default: false)
//
//    Currently, it does a simple weighted linear fit.

float EPICorrMultiBandSiemensSimpleGadget::filter_nav_correction_parameter( hoNDArray<float>& nav_corr_param_array,
                                                                   hoNDArray<float>& weights_array,
                                                                   size_t exc,
                                                                   size_t set,
                                                                   size_t slc,
                                                                   size_t Nt,
                                                                   bool   filter_in_complex_domain )
{
    // If the array to be filtered doesn't have 3 dimensions, we are in big trouble:
    if ( nav_corr_param_array.get_number_of_dimensions() != 3 )
    {
        GERROR("cbi_EPICorrMultiBandSiemensSimpleGadget::filter_nav_correction_parameter, incorrect number of dimensions of the array.\n");
        return -1;
    }

    // The dimensions of the weights array should be the same as the parameter array:
    if ( !nav_corr_param_array.dimensions_equal( &weights_array ) )
    {
        GERROR("cbi_EPICorrMultiBandSiemensSimpleGadget::filter_nav_correction_parameter, dimensions of the parameter and weights arrays don't match.\n");
        return -1;
    }

    // If this repetition number is less than then number of repetitions to exclude...
    if ( exc < navigatorParameterFilterExcludeVols.value()*E2_ )
    {
        //   no filtering is needed, just return the corresponding value:
        return nav_corr_param_array(exc, set, slc );
    }

    // for now, just to a simple (robust) linear fit to the previous Nt timepoints:
    // TO DO: do we want to do something fancier?

    //
    // extract the timeseries (e2 phase encoding steps and repetitions)
    // of parameters and weights corresponding to the requested indices:

    // make sure we don't use more timepoints (e2 phase encoding steps and repetitions)
    //    that the currently acquired (minus the ones we have been asked to exclude
    //    from the beginning of the run):
    Nt = std::min( Nt, exc - (navigatorParameterFilterExcludeVols.value()*E2_) + 1 );

    // create armadillo vectors, and stuff them in reverse order (from the
    // current timepoint, looking backwards). This way, the filtered value
    // we want would be simply the intercept):
    arma::fvec weights =  arma::zeros<arma::fvec>( Nt );
    arma::fvec params  =  arma::zeros<arma::fvec>( Nt );
    for (size_t t = 0; t < Nt; ++t)
    {
        weights(t) = weights_array(        exc-t, set, slc );
        params( t) = nav_corr_param_array( exc-t, set, slc );
    }

    /////     weighted fit:          b = (W*[1 t_])\(W*params);    /////

    float filtered_param;

    // if we need to filter in the complex domain:
    if (filter_in_complex_domain)
    {
        arma::cx_fvec zparams = arma::exp(arma::cx_fvec(arma::zeros<arma::fvec>( Nt ), params));            // zparams = exp( i*params );
        arma::cx_fvec B = arma::solve( arma::cx_fmat( arma::join_horiz( weights, weights % t_.head(Nt) ), arma::zeros<arma::fmat>( Nt,2) ),   weights % zparams );
        filtered_param = std::arg( arma::as_scalar(B(0)) );
    }
    else
    {
        arma::fvec B = arma::solve( arma::join_horiz( weights, weights % t_.head(Nt) ),  weights % params );
        filtered_param = arma::as_scalar(B(0));
    }

    //if ( exc==(weights_array.get_size(0)-1) && set==(weights_array.get_size(1)-1) &&
    //	 slc==(weights_array.get_size(2)-1) )
    //{
    //    write_nd_array< float >( &weights_array, "/tmp/nav_weights.real" );
    //    write_nd_array< float >( &nav_corr_param_array, "/tmp/nav_param_array.real" );
    //}
    //GDEBUG_STREAM("orig parameter: " << nav_corr_param_array(exc, set, slc) << "; filtered: " << filtered_param );

    return filtered_param;
}


////////////////////////////////////////////////////
//
//  increase_no_repetitions
//
//    funtion to increase the size of the navigator parameter arrays used for filtering
//    - delta_rep: how many more repetitions to add

void EPICorrMultiBandSiemensSimpleGadget::increase_no_repetitions( size_t delta_rep )
{

    GDEBUG_STREAM("cbi_EPICorrMultiBandSiemensSimpleGadget WARNING: repetition number larger than what specified in header");

    size_t REP     = Nav_mag_.get_size(0)/E2_;   // current maximum number of repetitions
    size_t new_REP = REP + delta_rep;
    size_t SET     = Nav_mag_.get_size(1);
    size_t SLC     = Nav_mag_.get_size(2);

    // create a new temporary array:
    hoNDArray<float> tmpArray( E2_*new_REP, SET, SLC);
    tmpArray.fill(float(0.));

    // For each navigator parameter array, copy what we have so far to the temporary array, and then copy back:

    // Nav_mag_ :
    for (size_t slc = 0; slc < SLC; ++slc)
    {
        for (size_t set = 0; set < SET; ++set)
        {
            memcpy( &tmpArray(0,set,slc), &Nav_mag_(0,set,slc), Nav_mag_.get_number_of_bytes()/SET/SLC );
        }
    }
    Nav_mag_ = tmpArray;

    // B0_intercept_ :
    for (size_t slc = 0; slc < SLC; ++slc)
    {
        for (size_t set = 0; set < SET; ++set)
        {
            memcpy( &tmpArray(0,set,slc), &B0_intercept_(0,set,slc), B0_intercept_.get_number_of_bytes()/SET/SLC );
        }
    }
    B0_intercept_ = tmpArray;

    // B0_slope_ :
    if (B0CorrectionMode.value().compare("linear")==0)
    {
        for (size_t slc = 0; slc < SLC; ++slc)
        {
            for (size_t set = 0; set < SET; ++set)
            {
                memcpy( &tmpArray(0,set,slc), &B0_slope_(0,set,slc), B0_slope_.get_number_of_bytes()/SET/SLC );
            }
        }
        B0_slope_ = tmpArray;
    }

    // OE_phi_intercept_ :
    for (size_t slc = 0; slc < SLC; ++slc)
    {
        for (size_t set = 0; set < SET; ++set)
        {
            memcpy( &tmpArray(0,set,slc), &OE_phi_intercept_(0,set,slc), OE_phi_intercept_.get_number_of_bytes()/SET/SLC );
        }
    }
    OE_phi_intercept_ = tmpArray;

    // OE_phi_slope_ :
    if ((OEPhaseCorrectionMode.value().compare("linear")==0    ) ||
            (OEPhaseCorrectionMode.value().compare("polynomial")==0) )
    {
        for (size_t slc = 0; slc < SLC; ++slc)
        {
            for (size_t set = 0; set < SET; ++set)
            {
                memcpy( &tmpArray(0,set,slc), &OE_phi_slope_(0,set,slc), OE_phi_slope_.get_number_of_bytes()/SET/SLC );
            }
        }
        OE_phi_slope_ = tmpArray;

        // OE_phi_poly_coef_ :
        if (OEPhaseCorrectionMode.value().compare("polynomial")==0)
        {
            for (size_t i = 0; i < OE_phi_poly_coef_.size(); ++i)
            {
                for (size_t slc = 0; slc < SLC; ++slc)
                {
                    for (size_t set = 0; set < SET; ++set)
                    {
                        memcpy( &tmpArray(0,set,slc), &OE_phi_poly_coef_[i](0,set,slc), OE_phi_poly_coef_[i].get_number_of_bytes()/SET/SLC );
                    }
                }
                OE_phi_poly_coef_[i] = tmpArray;
            }
        }
    }

}


int EPICorrMultiBandSiemensSimpleGadget::get_stack_number_from_gt_numerotation(int slice)
{

    arma::uvec q1 = arma::find(vec_MapSliceSMS == order_of_acquisition_sb(slice));

    int value= arma::conv_to<int>::from(q1);

    int stack_number= (value-(value%MB_factor_))/MB_factor_;

    return stack_number;
}


int EPICorrMultiBandSiemensSimpleGadget::get_stack_number_from_spatial_numerotation(int slice)
{

    arma::uvec q1 = arma::find(vec_MapSliceSMS == slice);

    int value= arma::conv_to<int>::from(q1);

    int stack_number= (value-(value%MB_factor_))/MB_factor_;

    return stack_number;
}



int EPICorrMultiBandSiemensSimpleGadget::get_blipped_factor(int numero_de_slice)
{

    int value=(numero_de_slice-(numero_de_slice%lNumberOfStacks_))/lNumberOfStacks_;

    return value;
}



void EPICorrMultiBandSiemensSimpleGadget::get_multiband_parameters(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec store_info_special_card=Gadgetron::get_information_from_wip_multiband_special_card(h);

    MB_factor_=store_info_special_card(1);
    Blipped_CAIPI_=store_info_special_card(4);
}

int EPICorrMultiBandSiemensSimpleGadget::get_spatial_slice_number(int slice)
{
    int value=order_of_acquisition_sb(slice);

    return value;
}





int EPICorrMultiBandSiemensSimpleGadget::transfer_kspace_to_the_next_gadget_2(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2 , unsigned int index)
{

    // ici il faudrait enlever indice_sb
    // il faut cependant vérifier que get_stack_number_from_gt_numerotation reste correcte

    GDEBUG("STEP 4 :Sending back slice : %d  , indice: %d  \n",    index , index );

    for (unsigned long i = 0; i < e1_size_for_epi_calculation; i++)
    {
        if(matrix_encoding(i,indice_sb(index))>0)  // ce flag sert a ne pas envoyer les lignes vide dans le cas du grappa
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

            arma::cx_fmat input= slice_calibration.slice(index).col(i);

            arma::cx_fmat output = reshape( input, readout, lNumberOfChannels_ );

             // ce flag sert a differencier les lignes pair et impair
            if (reverse_encoding(i,indice_sb(index))>0)
            {
                m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE);
                // Negative readout
                for (int p=0; p<output.n_cols; p++) {
                    //adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrneg_);
                    output.col(p) %=  corrneg_mean_.col(get_stack_number_from_gt_numerotation(index));
                    //output.col(p) %=  corrneg_all_.col(indice_sb(slice));
                }
                // Now that we have corrected we set the readout direction to positive
            }
            else {
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE);
                // Positive readout
                for (int p=0; p<output.n_cols; p++) {
                    //adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrpos_);
                    output.col(p) %=  corrpos_mean_.col(get_stack_number_from_gt_numerotation(index));
                    //output.col(p) %=  corrpos_all_.col(indice_sb(slice));
                }
            }

            std::complex<float>* dst = cm2->getObjectPtr()->get_data_ptr();

            for (unsigned long j = 0; j < m2->getObjectPtr()->get_number_of_elements(); j++)
            {
                dst[j]=output[j];
            }

            if (i==flag_encoding_first_scan_in_slice) //TODO
            {
               cm1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
            }
            else
            {
               cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
            }

            if (i==flag_encoding_last_scan_in_slice)  //TODO
            {
                cm1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
            }
            else
            {
                cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
            }

            if( (index==(lNumberOfSlices_-1)) && (i==flag_encoding_last_scan_in_slice))
            {
                GDEBUG("STEP 4 :Sending back slice : %d  %d  ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION \n",index ,lNumberOfSlices_-1 );
                cm1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION);
            }
            else
            {
                cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION);
            }

            //
            cm1->getObjectPtr()->idx.slice = index ;
            //cm1->getObjectPtr()->idx.slice = index ;
            cm1->getObjectPtr()->idx.kspace_encode_step_1=i;

            cm1->cont(cm2);

            if( this->next()->putq(cm1) < 0 ){
                GDEBUG("Failed to put message on queue\n");
                return GADGET_FAIL;
            }
        }
    }

return 0;

}


void EPICorrMultiBandSiemensSimpleGadget::detect_flag(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1)
{

    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

    bool is_first_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_ENCODE_STEP1);
    bool is_last_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;


   /* if (is_first_in_encoding)
    {
        flag_encoding_first_scan_in_encoding=e1;
        std::cout<< "  flag_encoding_first_scan_in_encoding " << flag_encoding_first_scan_in_encoding  << std::endl;
    }

    if (is_last_in_encoding)
    {
        flag_encoding_last_scan_in_encoding=e1;
        std::cout<< "  flag_encoding_last_scan_in_encoding " << flag_encoding_last_scan_in_encoding  << std::endl;
    }*/

    if (is_first_scan_in_slice)
    {
        flag_encoding_first_scan_in_slice=e1;
        std::cout<< "  flag_encoding_first_scan_in_slice " << flag_encoding_first_scan_in_slice  << std::endl;
    }


    if (is_last_scan_in_slice)
    {
        flag_encoding_last_scan_in_slice=e1;
        std::cout<< "  flag_encoding_last_scan_in_slice " << flag_encoding_last_scan_in_slice  << std::endl;
    }

}


void EPICorrMultiBandSiemensSimpleGadget::find_encoding_dimension_for_epi_calculation(void)
{
    if(acceFactorE1_>1)
    {
        e1_size_for_epi_calculation=encoding;
    }
    else
    {
        e1_size_for_epi_calculation=encoding;
    }

    GDEBUG("Encoding size for sms calculation: %d\n", e1_size_for_epi_calculation);
}


void EPICorrMultiBandSiemensSimpleGadget::deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h)
{
    if (doOffline.value()==1)
    {
        MB_factor_=MB_factor.value();
        Blipped_CAIPI_=Blipped_CAIPI.value();
        //MB_Slice_Inc_=MB_Slice_Inc.value();
    }
    else
    {
        get_multiband_parameters(h);
    }
}


void EPICorrMultiBandSiemensSimpleGadget::fonction_qui_sauvegarde_en_memoire_et_somme_les_corrections(int slice, int Nx_local, arma::fvec tvec_local )
{

    // TODO la condition sur compteur_sb_sum doit être respecté
    // ne marche pas en présence d'average par ex ou de multi-echo

    // on sauvegarde par coupes
    corrpos_all_.col(slice) = corrpos_;
    corrneg_all_.col(slice) = corrneg_;

    // on sauvegarde aussi sans les exp
    corrpos_no_exp_save_.col(slice) = arma::cx_fvec(arma::zeros<arma::fvec>(Nx_local), -0.5*tvec_local);
    corrneg_no_exp_save_.col(slice) = arma::cx_fvec(arma::zeros<arma::fvec>(Nx_local), +0.5*tvec_local);

    // on somme ces corrections par stacks
    corrpos_no_exp_mean_.col(get_stack_number_from_gt_numerotation(slice))+=(corrpos_no_exp_save_.col(slice)/MB_factor_);
    corrneg_no_exp_mean_.col(get_stack_number_from_gt_numerotation(slice))+=(corrneg_no_exp_save_.col(slice)/MB_factor_);

    GDEBUG("STEP 1 : EPI correction : slice %d , compteur_sb_sum %d, order_of_acquisition_sb %d , stack number %d , blip value %d \n", slice , compteur_sb_sum,  order_of_acquisition_sb(slice), get_stack_number_from_gt_numerotation(slice) , get_blipped_factor(order_of_acquisition_sb(slice)) );

    // on peut le faire uniquement si on est au dernier stack
    //TODO le code ci dessous peut entrer dans la fonction qui sauvegarde
    //if (get_blipped_factor(order_of_acquisition_sb(slice))==(MB_factor_-1))

    if (compteur_sb_sum== lNumberOfSlices_)
    {
        GDEBUG("STEP 1 : EPI correction : END calculating the average ghost-niquist correction for each stack \n");

        for (int ss = 0; ss < lNumberOfSlices_; ss++)
        {
        corrpos_mean_.col(get_stack_number_from_gt_numerotation(ss))=arma::exp(corrpos_no_exp_mean_.col(get_stack_number_from_gt_numerotation(ss)));
        corrneg_mean_.col(get_stack_number_from_gt_numerotation(ss))=arma::exp(corrneg_no_exp_mean_.col(get_stack_number_from_gt_numerotation(ss)));
        }

        flag_average_ghost_niquist_correction_are_available=1;
    }


}


void EPICorrMultiBandSiemensSimpleGadget::fonction_qui_sauvegarde_sur_le_disk_les_corrections_par_coupes(int slice )
{
    GDEBUG("STEP 2 : EPI correction : slice %d , compteur_sb_sum %d  saving optimal ghost niquist correction \n", slice , compteur_sb_sum);

    std::ostringstream indice_slice;
    indice_slice << slice;
    str_s = indice_slice.str();

    //std::cout << " size(corrneg_) "<< size(corrneg_) << std::endl;
    //std::cout << " size(corrpos_) "<< size(corrpos_) << std::endl;

    Gadgetron::SaveVectorOntheDisk(corrneg_, "/tmp/", "gadgetron/", "corrneg_",   str_s,  ".bin");
    Gadgetron::SaveVectorOntheDisk(corrpos_, "/tmp/", "gadgetron/", "corrpos_",   str_s,  ".bin");

    //std::cout << " size(corrneg_no_exp_save_) "<< size(corrneg_no_exp_save_.col(slice)) << std::endl;
    //std::cout << " size(corrneg_no_exp_save_) "<< size(corrpos_no_exp_save_.col(slice)) << std::endl;

    Gadgetron::SaveVectorOntheDisk(corrneg_no_exp_save_.col(slice), "/tmp/", "gadgetron/", "corrneg_no_exp_",   str_s,  ".bin");
    Gadgetron::SaveVectorOntheDisk(corrpos_no_exp_save_.col(slice), "/tmp/", "gadgetron/", "corrpos_no_exp_",   str_s,  ".bin");

}

void EPICorrMultiBandSiemensSimpleGadget::fonction_qui_sauvegarde_sur_le_disk_les_corrections_par_stacks(int slice )
{

    // si on est a la derniere coupe des slices de calibration
    // attention avant c'etait Blip_Caipi
    //if (get_blipped_factor(order_of_acquisition_sb(slice))==(MB_factor_-1))
    if (compteur_sb_sum == lNumberOfSlices_)
    {
        GDEBUG("STEP 3 : -------------- : condition  compteur_sb_sum %d == lNumberOfSlices_ %d \n", compteur_sb_sum , lNumberOfSlices_);
        GDEBUG("STEP 3 : EPI correction : slice %d , compteur_sb_sum %d  saving average ghost niquist correction\n",   slice , compteur_sb_sum);


        for (int a = 0; a < lNumberOfStacks_; a++)
        {

            std::ostringstream indice_stack;
            indice_stack << get_stack_number_from_gt_numerotation(a);
            str_a = indice_stack.str();

             GDEBUG("STEP 3 : EPI correction : saving mean of EPI phase correction for stack n° %d, stack position  %d \n" ,a,    get_stack_number_from_gt_numerotation(a));

            //Gadgetron::SaveMatrixOntheDisk(corrneg_no_exp_mean_, str_home, "Tempo/", "corrneg_no_exp_mean_",   str_a,  ".bin");
            //Gadgetron::SaveMatrixOntheDisk(corrpos_no_exp_mean_, str_home, "Tempo/", "corrpos_no_exp_mean_",   str_a,  ".bin");

            Gadgetron::SaveVectorOntheDisk(corrneg_mean_.col(a), str_home, "Tempo/", "corrneg_mean_",   str_a,  ".bin");
            Gadgetron::SaveVectorOntheDisk(corrpos_mean_.col(a), str_home, "Tempo/", "corrpos_mean_",   str_a,  ".bin");

        }

    }

}


GADGET_FACTORY_DECLARE(EPICorrMultiBandSiemensSimpleGadget)
}

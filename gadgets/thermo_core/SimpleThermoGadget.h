/**
\file   SimpleThermoGadget.h
\brief  This is the class gadget for cardiac T2 mapping, working on the IsmrmrdImageArray.
\author Hui Xue
*/

#pragma once

#include "ThermoMappingBase.h"
#include "hoNDFFT.h"
#include "hoNDArray_math.h"
#include "gadgetron_thermocore_export.h"

namespace Gadgetron {

    class EXPORTGADGETSTHERMOCORE SimpleThermoGadget : public ThermoMappingBase
    {
    public:
        GADGET_DECLARE(SimpleThermoGadget);

        typedef ThermoMappingBase BaseClass;

        SimpleThermoGadget();
        ~SimpleThermoGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the mapping
        /// ------------------------------------------------------------------------------------
        GADGET_PROPERTY(reference_number, unsigned int, "reference_number", 0);
        GADGET_PROPERTY(use_average_phase_for_reference, bool, "average_phase_reference", false);
        GADGET_PROPERTY(event_heating_time, int, "event_heating_time", 10);


    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // --------------------------------------------------
        // functional functions
        // --------------------------------------------------

        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);

        void compute_mean_std(hoNDArray<float> & buffer_temperature);

        void PrintFigletPower (int powerInWatt, int decompte , int fin);

        hoNDArray<float > magnitude;
        hoNDArray<float > phase;
        hoNDArray<float > temperature;

        hoNDArray<float > sum_of_magnitude;
        hoNDArray<float > sum_of_phase;

        hoNDArray<float > reference_m;
        hoNDArray<float > reference_p;

        hoNDArray<float > buffer_avant_reference;
        hoNDArray<float > buffer_temperature;

        hoNDArray<float > buffer_magnitude_all;
        hoNDArray<float > buffer_phase_all;
        hoNDArray<float > buffer_temperature_all;



        // function to perform the mapping
        // data: input image array [RO E1 E2 CHA N S SLC]
        // map and map_sd: mapping result and its sd
        // para and para_sd: other parameters of mapping and its sd
        //virtual int perform_zerofilling(IsmrmrdImageArray& data, IsmrmrdImageArray& map);
        //virtual void perform_zerofilling(IsmrmrdImageArray& in, hoNDArray< std::complex<float> > & data_out);
        //virtual void perform_zerofilling(IsmrmrdImageArray& in, IsmrmrdImageArray& out);


  };
}

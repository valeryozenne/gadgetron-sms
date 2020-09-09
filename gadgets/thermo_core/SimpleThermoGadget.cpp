
#include "SimpleThermoGadget.h"
#include <iomanip>
#include <sstream>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"
#include "mri_core_utility.h"



namespace Gadgetron {

SimpleThermoGadget::SimpleThermoGadget() : BaseClass()
{
}

SimpleThermoGadget::~SimpleThermoGadget()
{
}

int SimpleThermoGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(ThermoMappingBase::process_config(mb) == GADGET_OK, GADGET_FAIL);

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

    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;



    // -------------------------------------------------

    return GADGET_OK;
}

int SimpleThermoGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
{
    if (perform_timing.value()) { gt_timer_local_.start("SimpleThermoGadget::process"); }

    GDEBUG_CONDITION_STREAM(verbose.value(), "SimpleThermoGadget::process(...) starts ... ");

    // -------------------------------------------------------------

    process_called_times_++;

    // -------------------------------------------------------------
    if (lNumberOfRepetitions_>1)
    {
        IsmrmrdImageArray* data = m1->getObjectPtr();

        hoNDArray< ISMRMRD::ImageHeader > headers_=m1->getObjectPtr()->headers_;

        size_t rep   = headers_(0, 0, 0).repetition;

        GDEBUG_STREAM("---->rep "<< rep);

        std::stringstream os;
        os << "_rep_" << rep ;


        hoNDArray<std::complex<float> >  tempo=data->data_;

        size_t RO = data->data_.get_size(0);
        size_t E1 = data->data_.get_size(1);
        size_t E2 = data->data_.get_size(2);
        size_t CHA = data->data_.get_size(3);
        size_t N = data->data_.get_size(4);
        size_t S = data->data_.get_size(5);
        size_t SLC = data->data_.get_size(6);

        if (rep==0)
        {
            reference_m.create(RO,E1,E2,CHA,N,S,SLC);
            reference_p.create(RO,E1,E2,CHA,N,S,SLC);

            magnitude.create(RO,E1,E2,CHA,N,S,SLC);
            phase.create(RO,E1,E2,CHA,N,S,SLC);
            temperature.create(RO,E1,E2,CHA,N,S,SLC);

            sum_of_magnitude.create(RO,E1,E2,CHA,N,S,SLC);
            sum_of_phase.create(RO,E1,E2,CHA,N,S,SLC);

            Gadgetron::clear(sum_of_magnitude);
            Gadgetron::clear(sum_of_phase);

            buffer_avant_reference.create(RO,E1,E2,CHA,N,S,SLC, reference_number.value()+1);
            buffer_temperature.create(RO,E1,E2,CHA,N,S,SLC, lNumberOfRepetitions_- (reference_number.value()+1));

            buffer_temperature_all.create(RO,E1,E2,CHA,N,SLC, lNumberOfRepetitions_);
            buffer_magnitude_all.create(RO,E1,E2,CHA,N,SLC, lNumberOfRepetitions_);
            buffer_phase_all.create(RO,E1,E2,CHA,N,SLC, lNumberOfRepetitions_);

            std::cout << "reference_number "<< reference_number.value()<< std::endl;
            std::cout << "reference_number +1 "<< reference_number.value() +1 << std::endl;
            std::cout << "lNumberOfRepetitions_ "<< lNumberOfRepetitions_<< std::endl;
            std::cout << "lNumberOfInterventions  "<< lNumberOfRepetitions_- (reference_number.value()+1)<< std::endl;
            std::cout << "reference_number "<< reference_number<< std::endl;

        }

        Gadgetron::abs(data->data_, magnitude);
        phase=Gadgetron::argument(data->data_);

        // soit creer trois nouveaux messages
        // soit travailler dans ce gadget
        // soit envoyer juste en plus la température à retro control
        // utiliser le tag data role

        memcpy(&buffer_magnitude_all(0,0,0,0,0,0,rep), &magnitude(0,0,0,0,0,0,0), sizeof(float)*RO*E1*E2*CHA*N*SLC);
        memcpy(&buffer_phase_all(0,0,0,0,0,0,rep), &phase(0,0,0,0,0,0,0), sizeof(float)*RO*E1*E2*CHA*N*SLC);

        /*if (!debug_folder_full_path_.empty()) {


        this->gt_exporter_.export_array(magnitude,
                                        debug_folder_full_path_ + "magnitude"+ os.str());

        this->gt_exporter_.export_array(phase,
                                        debug_folder_full_path_ + "phase"+ os.str());
    }*/

        if (rep<reference_number.value())
        {
            Gadgetron::add(sum_of_magnitude , magnitude, sum_of_magnitude ) ;
            Gadgetron::add(sum_of_phase , phase, sum_of_phase ) ;

        }
        else if (rep==reference_number.value())
        {
            reference_m=magnitude;

            if (use_average_phase_for_reference.value() )
            {
                Gadgetron::scal(1/(float)(reference_number.value()+1), sum_of_phase);
                reference_p=sum_of_phase;
            }
            else
            {
                reference_p=phase;
            }


            Gadgetron::add(sum_of_magnitude , magnitude, sum_of_magnitude ) ;

            GDEBUG("--------------- %d  %d ", reference_number.value()+1 , process_called_times_  );

            hoNDArray<float>mean_of_magnitude(sum_of_magnitude);
            Gadgetron::scal(1/(float)(reference_number.value()+1), mean_of_magnitude);

            if (!debug_folder_full_path_.empty()) {

                this->gt_exporter_.export_array( mean_of_magnitude,
                                                 debug_folder_full_path_ + "sum_of_magnitude");

                this->gt_exporter_.export_array( reference_p,
                                                 debug_folder_full_path_ + "reference_p");

            }

            //retro-processing
        }
        else if (rep>reference_number.value())
        {


            if (rep<event_heating_time.value())
            {
                PrintFigletPower(0 , event_heating_time.value()-(rep+1) , rep+1);
            }

            size_t idx_intervention=rep-(reference_number.value()+1);

            Gadgetron::add(sum_of_magnitude , magnitude, sum_of_magnitude ) ;
            Gadgetron::subtract(phase, reference_p, temperature);
            Gadgetron::scal(k_value_, temperature);

            /*if (!debug_folder_full_path_.empty()) {
                this->gt_exporter_.export_array(temperature,
                                                debug_folder_full_path_ + "temperature"+ os.str());

            }*/

            memcpy(&buffer_temperature(0,0,0,0,0,0,0,idx_intervention), &temperature(0,0,0,0,0,0,0), sizeof(float)*RO*E1*E2*CHA*N*S*SLC);
            memcpy(&buffer_temperature_all(0,0,0,0,0,0,rep), &temperature(0,0,0,0,0,0,0), sizeof(float)*RO*E1*E2*CHA*N*S*SLC);


            if (rep==100-1)
            {
                GDEBUG("--------repetition 200-----\n");

                //compute_mean_std(buffer_temperature);


                std::stringstream os;
                buffer_magnitude_all.print(os);
                GDEBUG_STREAM(os.str());

                std::stringstream os2;
                buffer_phase_all.print(os2);
                GDEBUG_STREAM(os2.str());

                std::stringstream os3;
                buffer_temperature_all.print(os3);
                GDEBUG_STREAM(os3.str());

                if (!debug_folder_full_path_.empty()) {
                    this->gt_exporter_.export_array(buffer_magnitude_all,
                                                    debug_folder_full_path_ + "buffer_magnitude_200");    }

                if (!debug_folder_full_path_.empty()) {
                    this->gt_exporter_.export_array(buffer_phase_all,
                                                    debug_folder_full_path_ + "buffer_phase_200");      }

                if (!debug_folder_full_path_.empty()) {
                    this->gt_exporter_.export_array(buffer_temperature_all,
                                                    debug_folder_full_path_ + "buffer_temperature_200");        }

            }


            if (rep==lNumberOfRepetitions_-1)
            {
                GDEBUG("--------Last repetition -----\n");

                compute_mean_std(buffer_temperature);


                std::stringstream os;
                buffer_magnitude_all.print(os);
                GDEBUG_STREAM(os.str());

                std::stringstream os2;
                buffer_phase_all.print(os2);
                GDEBUG_STREAM(os2.str());

                std::stringstream os3;
                buffer_temperature_all.print(os3);
                GDEBUG_STREAM(os3.str());

                if (!debug_folder_full_path_.empty()) {
                    this->gt_exporter_.export_array(buffer_magnitude_all,
                                                    debug_folder_full_path_ + "buffer_magnitude_all");    }

                if (!debug_folder_full_path_.empty()) {
                    this->gt_exporter_.export_array(buffer_phase_all,
                                                    debug_folder_full_path_ + "buffer_phase_all");      }

                if (!debug_folder_full_path_.empty()) {
                    this->gt_exporter_.export_array(buffer_temperature_all,
                                                    debug_folder_full_path_ + "buffer_temperature_all");        }

            }

            // do baseline correction
            // do filtering

        }



        /*
    if (verbose.value())
    {
        GDEBUG_STREAM("----> SimpleThermoGadget::process(...) has been called " << process_called_times_ << " times ...");
        std::stringstream os;
        data->data_.print(os);
        GDEBUG_STREAM(os.str());
    }*/


    }
    /*Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>* cm2 = new Gadgetron::GadgetContainerMessage<IsmrmrdImageArray>();
    IsmrmrdImageArray map_sd;

    //M, P, T
    map_sd.data_.create(RO, E1, E2, CHA, N, S, SLC,3);
    map_sd.data_.fill(10);
    map_sd.headers_.create(N, S, SLC);
    map_sd.meta_.resize(N*S*SLC);

    *(cm2->getObjectPtr()) = map_sd;

    m1->release();*/

    // -------------------------------------------------------------

    if (this->next()->putq(m1) == -1)
    {
        GERROR("SimpleThermoGadget::process, passing map on to next gadget");
        return GADGET_FAIL;
    }

    if (perform_timing.value()) { gt_timer_local_.stop(); }

    return GADGET_OK;
}

void SimpleThermoGadget::compute_mean_std(hoNDArray<float> & buffer_temperature)
{
    size_t RO = buffer_temperature.get_size(0);
    size_t E1 = buffer_temperature.get_size(1);
    size_t E2 = buffer_temperature.get_size(2);
    size_t CHA = buffer_temperature.get_size(3);
    size_t N = buffer_temperature.get_size(4);
    size_t S = buffer_temperature.get_size(5);
    size_t SLC = buffer_temperature.get_size(6);
    size_t T = buffer_temperature.get_size(7);

    GDEBUG_STREAM( " RO :  "  << RO  <<" E1: "<< E1 <<  " E2 :  "  << E2  <<" CHA: "<< CHA <<  " N :  "  << N  <<" S: "<< S <<" SLC: " << SLC<<" T: " << T);

    hoNDArray<float>  temperature_mean(RO,E1,E2,CHA,N,S,SLC);

    hoNDArray<float>  temperature_std(RO,E1,E2,CHA,N,S,SLC);

    size_t t,slc,n,s,cha,e2,e1,ro;

    hoNDArray<float> buf(T);

    for (long long slc = 0; slc < SLC; slc++)
    {
        for (long long s = 0; s < S; s++)
        {
            for (long long n = 0; n < N; n++)
            {
                for (long long cha = 0; cha < CHA; cha++)
                {
                    for (long long e2 = 0; e2 < E2; e2++)
                    {
                        for (long long e1 = 0; e1 < E1; e1++)
                        {
                            for (long long ro = 0; ro < RO; ro++)
                            {
                                Gadgetron::clear(buf);

                                for (long long t = 0; t < T; t++)
                                {
                                    buf(t)=buffer_temperature(ro,e1,e2,cha,n,s,slc,t);
                                }

                                temperature_std(ro,e1,e2,cha,n,s,slc)=Gadgetron::stddev(&buf);
                                temperature_mean(ro,e1,e2,cha,n,s,slc)=Gadgetron::mean(&buf);

                            }
                        }
                    }
                }
            }
        }
    }

    if (!debug_folder_full_path_.empty()) {
        this->gt_exporter_.export_array(temperature_std,
                                        debug_folder_full_path_ + "temperature_std");

    }

    if (!debug_folder_full_path_.empty()) {
        this->gt_exporter_.export_array(temperature_mean,
                                        debug_folder_full_path_ + "temperature_mean");

    }


}

// ----------------------------------------------------------------------------------------


void SimpleThermoGadget::PrintFigletPower (int powerInWatt, int decompte , int fin)
{

    std::ostringstream affichage_decompte;
    affichage_decompte << decompte;

    std::ostringstream affichage_fin;
    affichage_fin << fin;

    std::ostringstream affichage_power;
    affichage_power << powerInWatt;

    std::string str_temp = "figlet -c " + affichage_power.str() + " W "  + affichage_decompte.str() + " / " + affichage_fin.str() ;
    const char * c_affichage=str_temp.c_str();

    //std::cout <<  c_affichage << std::endl;

    bool status=system(c_affichage);
    std::cout <<  "\n" << std::endl;

}

GADGET_FACTORY_DECLARE(SimpleThermoGadget)

}

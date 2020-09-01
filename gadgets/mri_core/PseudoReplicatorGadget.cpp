/*
 * PseudoReplicator.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: u051747
 */

#include "PseudoReplicatorGadget.h"

#include <random>
namespace Gadgetron {

PseudoReplicatorGadget::PseudoReplicatorGadget() : Gadget1<IsmrmrdReconData>() {
    // TODO Auto-generated constructor stub

}

PseudoReplicatorGadget::~PseudoReplicatorGadget() {
    // TODO Auto-generated destructor stub
}

int PseudoReplicatorGadget::process_config(ACE_Message_Block*) {

    repetitions_ = repetitions.value();

    if (!debug_folder.value().empty())
    {
        Gadgetron::get_debug_folder_path(debug_folder.value(), debug_folder_full_path_);
        GDEBUG_CONDITION_STREAM(verbose.value(), "Debug folder is " << debug_folder_full_path_);

        // Create debug folder if necessary
        boost::filesystem::path boost_folder_path(debug_folder_full_path_);
        try
        {
            boost::filesystem::create_directories(boost_folder_path);
        }
        catch (...)
        {
            GERROR("Error creating the debug folder.\n");
            return false;
        }
    }
    else
    {
        GDEBUG_CONDITION_STREAM(verbose.value(), "Debug folder is not set ... ");
    }

    debug_folder_full_path_network_="/home/valery/Reseau/Imagerie/For_Quentin/G-Factor/Output/";

    return GADGET_OK;
}

int PseudoReplicatorGadget::process(GadgetContainerMessage<IsmrmrdReconData>* m) {

    std::mt19937 engine;
    std::normal_distribution<float> distribution; //{5,2};

    auto m_copy = *m->getObjectPtr();
    //First just send the normal data to obtain standard image
    if (this->next()->putq(m) == GADGET_FAIL)
        return GADGET_FAIL;

    //Now for the noisy projections
    for (int i =0; i < repetitions_; i++){

        std::stringstream os;
        os << "_repetition_" << i;

        std::cout << "coucou "<< i <<"/" <<  repetitions_ << std::endl;

        auto cm = new GadgetContainerMessage<IsmrmrdReconData>();
        *cm->getObjectPtr() = m_copy;
        auto & datasets = cm->getObjectPtr()->rbit_;

        for (auto & buffer : datasets){
            auto & data = buffer.data_.data_;
            auto & noise = buffer.data_.data_;

            gt_exporter_.export_array_complex(data, debug_folder_full_path_network_ + "data_before" + os.str());

            auto dataptr = data.get_data_ptr();
            auto noiseptr = noise.get_data_ptr();
            for (size_t k =0; k <  data.get_number_of_elements(); k++){
                dataptr[k] += std::complex<float>(distribution(engine),distribution(engine));
                noiseptr[k] = std::complex<float>(distribution(engine),distribution(engine));
            }

            gt_exporter_.export_array_complex(data, debug_folder_full_path_network_ + "noise" + os.str());
            gt_exporter_.export_array_complex(noise, debug_folder_full_path_network_ + "data_after" + os.str());
        }


        GDEBUG("Sending out Pseudoreplica\n");

        if (this->next()->putq(cm) == GADGET_FAIL)
            return GADGET_FAIL;

    }
    return GADGET_OK;

}

GADGET_FACTORY_DECLARE(PseudoReplicatorGadget)

} /* namespace Gadgetron */

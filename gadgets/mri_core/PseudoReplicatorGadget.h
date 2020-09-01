/*
 * PseudoReplicator.h

 *
 *  Created on: Jun 23, 2015
 *      Author: David Hansen
 */
#pragma once
#include "Gadget.h"
#include "mri_core_data.h"
#include "gadgetron_mricore_export.h"
#include "ImageIOAnalyze.h"
#include "mri_core_utility.h"

namespace Gadgetron {

class EXPORTGADGETSMRICORE PseudoReplicatorGadget : public Gadget1<IsmrmrdReconData>{
public:
	GADGET_PROPERTY(repetitions,int,"Number of pseudoreplicas to produce",10);
	PseudoReplicatorGadget()  ;
	virtual ~PseudoReplicatorGadget();

    Gadgetron::ImageIOAnalyze gt_exporter_;

    GADGET_PROPERTY(debug_folder, std::string, "If set, the debug output will be written out", "");
    GADGET_PROPERTY(verbose, bool, "Whether to print more information", false);

    std::string debug_folder_full_path_;
    std::string debug_folder_full_path_network_;

	virtual int process_config(ACE_Message_Block *);
	virtual int process(GadgetContainerMessage<IsmrmrdReconData>*);

private:
	int repetitions_;
};

} /* namespace Gadgetron */

/*
 * ExtractMagnitudeGadget.cpp
 *
 *  Created on: Nov 8, 2011
 *      Author: Michael S. Hansen
 */

//#include "GadgetIsmrmrdReadWrite.h"
#include "ExtractThermoIGadget.h"
#include "mri_core_def.h"
//#include "mri_core_utility_interventional.h"


namespace Gadgetron{
ExtractThermoIGadget::ExtractThermoIGadget()
    : extract_mask_(GADGET_EXTRACT_MAGNITUDE)
{

}

ExtractThermoIGadget::~ExtractThermoIGadget()
{

}

/// Get parameters from XML file
int ExtractThermoIGadget::process_config(ACE_Message_Block* mb) 
{

    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);


return GADGET_OK;

}  



int ExtractThermoIGadget::process(GadgetContainerMessage<ISMRMRD::ImageHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    if (doOffline.value()==1)
    {
        int em = extract_mask.value();

        if (em > 0) {
            if (em < GADGET_EXTRACT_MAX ) {
                extract_mask_ = static_cast<unsigned short>(em);
               // cout << " extract_mask_ " <<  extract_mask_ << endl;
            }
        }
    }
    else
    {
        extract_mask_ = static_cast<unsigned short>(doSendBackWhichImages_);
       // cout << " extract_mask_ " <<  extract_mask_ << endl;
    }

    static int counter = 0;
    for (size_t m = GADGET_EXTRACT_MAGNITUDE; m < GADGET_EXTRACT_MAX; m = m<<1) {
        if (extract_mask_ & m) {
            GadgetContainerMessage<ISMRMRD::ImageHeader>* cm1 =		new GadgetContainerMessage<ISMRMRD::ImageHeader>();
            Gadgetron::GadgetContainerMessage<ISMRMRD::MetaContainer>* cm3 = new Gadgetron::GadgetContainerMessage<ISMRMRD::MetaContainer>();

            //Copy the header
            *cm1->getObjectPtr() = *m1->getObjectPtr();

            //IsmrmrdImageArray* recon_res_ = m1->getObjectPtr();

            GadgetContainerMessage<hoNDArray< float > > *cm2 =	new GadgetContainerMessage<hoNDArray< float > >();

            boost::shared_ptr< std::vector<size_t> > dims = m2->getObjectPtr()->get_dimensions();

            try{cm2->getObjectPtr()->create(dims.get());}
            catch (std::runtime_error &err){
                GEXCEPTION(err,"Unable to create unsigned short storage in Extract Magnitude Gadget");
                return GADGET_FAIL;
            }

            std::complex<float>* src = m2->getObjectPtr()->get_data_ptr();
            float* dst = cm2->getObjectPtr()->get_data_ptr();

            float pix_val;
            for (unsigned long i = 0; i < cm2->getObjectPtr()->get_number_of_elements(); i++) {
                switch (m) {
                case GADGET_EXTRACT_MAGNITUDE:
                    pix_val = abs(src[i]);
                    break;
                case GADGET_EXTRACT_REAL:
                    pix_val = real(src[i]);
                    break;
                case GADGET_EXTRACT_IMAG:
                    pix_val = imag(src[i]);
                    break;
                case GADGET_EXTRACT_PHASE:
                    pix_val = arg(src[i]);
                    break;
                case GADGET_EXTRACT_THERMO:
                    pix_val = arg(src[i]);
                    break;
                default:
                    GDEBUG("Unexpected extract mask %d, bailing out\n", m);
                    return GADGET_FAIL;
                }
                dst[i] = pix_val;
            }

            cm1->cont(cm2);
            cm2->cont(cm3);
            cm1->getObjectPtr()->data_type = ISMRMRD::ISMRMRD_FLOAT;//GADGET_IMAGE_REAL_FLOAT;

            switch (m) {
            case GADGET_EXTRACT_MAGNITUDE:
                cm1->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;//GADGET_IMAGE_MAGNITUDE;
                cm3->getObjectPtr()->set(GADGETRON_SEQUENCEDESCRIPTION, "Magn");
                cm3->getObjectPtr()->set(GADGETRON_IMAGEPROCESSINGHISTORY, "GT");
                cm3->getObjectPtr()->set(GADGETRON_IMAGECOMMENT, "Bordeaux Thermometry");
                //cm3->getObjectPtr()->set(GADGETRON_SRC_STREAM_LAYER,(long)(0));
                //bool is_src_end=false;
                //cm3->getObjectPtr()->set(GADGETRON_SRC_STREAM_END,is_src_end);
                break;
            case GADGET_EXTRACT_REAL:
                cm1->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_REAL;
                cm1->getObjectPtr()->image_series_index += 1000; //Ensure that this will go in a different series
                break;
            case GADGET_EXTRACT_IMAG:
                cm1->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_IMAG;
                cm1->getObjectPtr()->image_series_index += 2000; //Ensure that this will go in a different series
                break;
            case GADGET_EXTRACT_PHASE:
                cm1->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_PHASE;
                cm1->getObjectPtr()->image_series_index += 3000; //Ensure that this will go in a different series
                cm3->getObjectPtr()->set(GADGETRON_SEQUENCEDESCRIPTION, "Phase");
                cm3->getObjectPtr()->set(GADGETRON_IMAGEPROCESSINGHISTORY, "GT");
                cm3->getObjectPtr()->set(GADGETRON_IMAGECOMMENT, "Bordeaux Thermometry");
                //cm3->getObjectPtr()->set(GADGETRON_SRC_STREAM_LAYER,(long)(1));
                //bool is_src_end=true;
                //cm3->getObjectPtr()->set(GADGETRON_SRC_STREAM_END,"true");
                break;
            /*case GADGET_EXTRACT_THERMO:
                cm1->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_THERMO;
                cm1->getObjectPtr()->image_series_index += 4000; //Ensure that this will go in a different series               
                cm3->getObjectPtr()->set(GADGETRON_IMAGE_SCALE_RATIO, (double)(1));
                cm3->getObjectPtr()->set(GADGETRON_IMAGECOMMENT, "Bordeaux Thermometry");
                cm3->getObjectPtr()->set(GADGETRON_SEQUENCEDESCRIPTION, "Temp");
                cm3->getObjectPtr()->set(GADGETRON_IMAGEPROCESSINGHISTORY, "GT");
                cm3->getObjectPtr()->set(GADGETRON_IMAGE_COLORMAP, "Perfusion.pal");
                cm3->getObjectPtr()->set(GADGETRON_IMAGE_WINDOWCENTER, (long)(38));
                cm3->getObjectPtr()->set(GADGETRON_IMAGE_WINDOWWIDTH, (long)(46));
                break;
                */
                default:
                GDEBUG("Unexpected extract mask %d, bailing out\n", m);
                break;
            }

            if (this->next()->putq(cm1) == -1) {
                m1->release();
                GDEBUG("Unable to put extracted images on next gadgets queue");
                return GADGET_FAIL;
            }
        }
    }

    m1->release(); //We have copied all the data in this case
    return GADGET_OK;
}






GADGET_FACTORY_DECLARE(ExtractThermoIGadget)
}

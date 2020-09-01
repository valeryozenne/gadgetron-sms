#include "ImageArrayEmptyGadget.h"
#include "mri_core_utility.h"
#include <math.h>
namespace Gadgetron{

ImageArrayEmptyGadget::ImageArrayEmptyGadget()
{

}

int ImageArrayEmptyGadget::process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1)
{
    if (this->next()->putq(m1) < 0)
    {
        m1->release();
        return GADGET_FAIL;
    }

    return GADGET_OK;
}

int ImageArrayEmptyGadget::process( GadgetContainerMessage<IsmrmrdImageArray>* m1)
{

    //Grab a reference to the buffer containing the imaging data
    IsmrmrdImageArray & imagearr = *m1->getObjectPtr();

    //7D, fixed order [X, Y, Z, CHA, N, S, LOC]
    uint16_t X = imagearr.data_.get_size(0);
    uint16_t Y = imagearr.data_.get_size(1);
    uint16_t Z = imagearr.data_.get_size(2);
    uint16_t CHA = imagearr.data_.get_size(3);
    uint16_t N = imagearr.data_.get_size(4);
    uint16_t S = imagearr.data_.get_size(5);
    uint16_t LOC = imagearr.data_.get_size(6);

    //Each image will be [X,Y,Z,CHA] big
    std::vector<size_t> img_dims(4);
    img_dims[0] = X;
    img_dims[1] = Y;
    img_dims[2] = Z;
    img_dims[3] = CHA;

    //Loop over N, S and LOC
    for (uint16_t loc=0; loc < LOC; loc++) {
        for (uint16_t s=0; s < S; s++) {
            for (uint16_t n=0; n < N; n++) {

                //Create a new image header and copy the header for this n, s and loc
                GadgetContainerMessage<ISMRMRD::ImageHeader>* cm1 =
                        new GadgetContainerMessage<ISMRMRD::ImageHeader>();
                memcpy(cm1->getObjectPtr(), &imagearr.headers_(n,s,loc), sizeof(ISMRMRD::ImageHeader));

                //Create a new image image
                // and the 4D data block [X,Y,Z,CHA] for this n, s and loc
                GadgetContainerMessage< hoNDArray< std::complex<float> > >* cm2 =
                        new GadgetContainerMessage<hoNDArray< std::complex<float> > >();

                try{cm2->getObjectPtr()->create(img_dims);}
                catch (std::runtime_error &err){
                    GEXCEPTION(err,"Unable to allocate new image\n");
                    cm1->release();
                    cm2->release();
                    return GADGET_FAIL;
                }
                memcpy(cm2->getObjectPtr()->get_data_ptr(), &imagearr.data_(0,0,0,0,n,s,loc), X*Y*Z*CHA*sizeof(std::complex<float>));
                //Chain them
                cm1->cont(cm2);


                //Creat a new meta container if needed and copy
                if (imagearr.meta_.size()>0) {

                    std::cout << "begin" <<std::endl;

                    GadgetContainerMessage< ISMRMRD::MetaContainer >* cm3 =
                            new GadgetContainerMessage< ISMRMRD::MetaContainer >();
                    size_t mindex = loc*N*S + s*N + n;

                    *cm3->getObjectPtr() = imagearr.meta_[mindex];

                    //coordinate in double as in IceGadgetron with 4 points



                    hoNDArray<double> endo(35,2);

                    int count=0;
                    for (int ll=0; ll < 360; ll+=10 )
                    {
                        endo(count,0) = 50 + 20* cos((double)ll/180*M_PI);
                        endo(count,1) = 50 + 20* sin((double)ll/180*M_PI);
                        std::cout <<count << " "<< ll<< " " << endo(count,0)<< " " <<  endo(count,1)<<std::endl;
                        count=count+1;
                    }

                    /*
                    hoNDArray<double> endo(5,2);

                    endo(0,0)=50;
                    endo(0,1)=50;

                    endo(1,0)=70;
                    endo(1,1)=50;

                    endo(2,0)=70;
                    endo(2,1)=70;

                    endo(3,0)=50;
                    endo(3,1)=70;

                    endo(4,0)=55;
                    endo(4,1)=75;*/


                    size_t n_endo = endo.get_size(0);

                    std::vector<double> endo_prepared;
                    endo_prepared.resize(2 * n_endo + 4, 0); // color + line width
                    endo_prepared[0] = 255; //color[0];
                    endo_prepared[1] = 0; //color[1];
                    endo_prepared[2] = 0; //color[2];
                    endo_prepared[3] = 1; //line_width;

                    for (size_t pt = 0; pt < n_endo; pt++)
                    {

                        endo_prepared[2 * pt + 4] = endo(pt, 0);
                        endo_prepared[2 * pt + 1 + 4] = endo(pt, 1);
                    }

                    std::string roi_label = GADGETRON_2D_ROI ; //+ std::string("_ENDO1");

                    std::vector<std::string> dataRole;

                    //typical modification that works
                    //cm3->getObjectPtr()->set(GADGETRON_IMAGE_COLORMAP, "Perfusion.pal");
                    cm3->getObjectPtr()->set(GADGETRON_IMAGECOMMENT, "Bordeaux Thermometry");
                    cm3->getObjectPtr()->set(GADGETRON_SEQUENCEDESCRIPTION, "Temperature ");
                    cm3->getObjectPtr()->set(GADGETRON_IMAGEPROCESSINGHISTORY, "GT");
                    //cm3->getObjectPtr()->set(GADGETRON_2D_ROI, "ENDO1");

                    // adding the following line make the image reco failed (no imae reception on siemens host)
                    //code compile and execute offline but image reconstruction failed from host side , logviewer tell me iimage reco failed and this in particular:
                    //"MRUIBackends_PAScontainer :  GetValue() failed for exam memory item MR AA results"
                    setISMRMRMetaValues(*cm3->getObjectPtr(), roi_label, endo_prepared);

                    std::vector<double> endo_prepared_check;

                    get_ismrmrd_meta_values(*cm3->getObjectPtr(), roi_label, endo_prepared_check);

                    for (size_t j = 0; j < endo_prepared_check.size(); j++)
                    {
                        std::cout << endo_prepared_check[j] << std::endl;
                    }

                    cm2->cont(cm3);

                    std::cout << "end" <<std::endl;
                }

                //Pass the image down the chain
                if (this->next()->putq(cm1) < 0) {
                    return GADGET_FAIL;
                }
            }
        }
    }

    m1->release();
    return GADGET_OK;

}



bool ImageArrayEmptyGadget::setISMRMRMetaValues(ISMRMRD::MetaContainer& attrib, const std::string& name, const std::vector<double>& v)
{
    try{

        size_t num = v.size();

        if (num == 0)
        {

            // GWARN_STREAM("setISMRMRMetaValues, input vector is empty ... " << name);
            return true;

        }

        attrib.set(name.c_str(), v[0]);

        size_t ii;

        for (ii = 1; ii < v.size(); ii++)
        {

            attrib.append(name.c_str(), v[ii]);

        }

    }
    catch (...)
    {
        GERROR_STREAM("Error happened in setISMRMRMetaValues(ISMRMRD::MetaContainer& attrib, const std::string& name, const std::vector<T>& v) ... ");

        return false;

    }

    return true;

}


GADGET_FACTORY_DECLARE(ImageArrayEmptyGadget)
}




import numpy as np
from ismrmrdtools import transform
from gadgetron import Gadget,IsmrmrdDataBuffered, IsmrmrdReconBit, SamplingLimit,SamplingDescription, IsmrmrdImageArray
import sys
import ismrmrd
import ismrmrd.xsd
from pygrappa import grappa

class BucketReconMP2RAGE(Gadget):
    def __init__(self, next_gadget=None):
        Gadget.__init__(self,next_gadget)
        self.array_calib=[]
        pass         

    def process_config(self, conf):
        pass      

    def process(self, recondata):

        # receive kspace data and
        # extract acq_data and acq_header
        array_acq_headers=recondata[0].data.headers
        
        kspace_data=recondata[0].data.data
        
        # get one header with typical info
        acq = np.ravel(array_acq_headers)[0]  

        print("acq.idx.repetition", acq.idx.repetition)        


        try: 
           if recondata[0].ref.data is not None:
             print("reference data exist")
             print(np.shape(recondata[0].ref.data)) # only for repetition 0 # il faut creer le bucket recon grappa
             reference=recondata[0].ref.data
             data=recondata[0].data.data
             print(np.shape(reference))
             print(np.shape(data))
             self.array_calib=reference
             np.save('/home/valery/DICOM/mp2rage_reference', reference)
             np.save('/home/valery/DICOM/mp2rage_data', data)
        except:
           print("reference data not exist")
        

        # grappa 
        # array_data=recondata[0].data.data
        # dims=np.shape(recondata[0].data.data)
       
        # kspace_data_tmp=np.ndarray(dims, dtype=np.complex64)   

        image = transform.transform_kspace_to_image(kspace_data,dim=(0,1,2))     

        # create a new IsmrmrdImageArray 
        array_data = IsmrmrdImageArray()
       
        # attache the images to the IsmrmrdImageArray      
        array_data.data=image

        # get dimension for the acq_headers
        dims_header=np.shape(recondata[0].data.headers)

       

        if (acq.idx.repetition==0):
           np.save('/home/valery/DICOM/mp2rage_image',image)


        headers_list = []
        base_header=ismrmrd.ImageHeader()
        base_header.version=2
        ndims_image=np.shape(image)
        base_header.channels = ndims_image[3]       
        base_header.matrix_size = (image.shape[0],image.shape[1],image.shape[2])
        print((image.shape[0],image.shape[1],image.shape[2]))
        base_header.position = acq.position
        base_header.read_dir = acq.read_dir
        base_header.phase_dir = acq.phase_dir
        base_header.slice_dir = acq.slice_dir
        base_header.patient_table_position = acq.patient_table_position
        base_header.acquisition_time_stamp = acq.acquisition_time_stamp
        base_header.image_index = 0 
        base_header.image_series_index = 0
        base_header.data_type = ismrmrd.DATATYPE_CXFLOAT
        base_header.image_type= ismrmrd.IMTYPE_MAGNITUDE
        print("ready to list")

        for slc in range(0, dims_header[4]):
           for n in range(0, dims_header[3]):
              for s in range(0, dims_header[2]):
                 #for e2 in range(0, dims_header[1]):
                 #  for e1 in range(0, dims_header[0]):  
                     headers_list.append(base_header)

        
        array_headers_test = np.array(headers_list,dtype=np.dtype(object))  
        print(type(array_headers_test))
        print(np.shape(array_headers_test))
        array_headers_test=np.reshape(array_headers_test, (dims_header[2], dims_header[3], dims_header[4])) 
        print(type(array_headers_test))
        print(np.shape(array_headers_test))
 
        
           
        print("---> ok 0")
        # how to copy acquisition header into image header in python  ? 
        for slc in range(0, dims_header[4]):
           for n in range(0, dims_header[3]):
              for s in range(0, dims_header[2]):
                # for e2 in range(0, dims_header[1]):
                  # for e1 in range(0, dims_header[0]):
                        array_headers_test[s,n,slc].slice=slc
                        #print(s,n,slc)      
                        #print(type(array_headers_test[s,n,slc]))
                        #print(array_headers_test[s,n,slc].slice)
                                                

        #print("---> ok 1")
        # print(np.shape(array_image_header))      
        # attache the image headers to the IsmrmrdImageArray      
        array_data.headers=array_headers_test

        #print("---> ok 2")
        # Return image to Gadgetron
        #print(np.shape(array_data.data))
        #print(np.shape(array_data.headers))

        #print(type(array_data.data))
        #print(type(array_data.headers))
        #print(type(array_data))

	# send the data to the next gadget

        for slc in range(0, dims_header[4]):
           for n in range(0, dims_header[3]):
              for s in range(0, dims_header[2]):
                    #print("send out image %d-%d-%d" % (s, n, slc))
                    a = array_data.data[:,:,:,:,s,n,slc]
                    #array_data.headers[s,n,slc].slice=slc
                    #print(a.shape, array_data.headers[s,n,slc].slice)
                    self.put_next(array_data.headers[s,n,slc], a)

        #self.put_next( [IsmrmrdReconBit(array_data.headers, array_data.data)] ,  )  
      
        print("----------------------------------------------")
        return 0    
        # print "Returning to Gadgetron"
 

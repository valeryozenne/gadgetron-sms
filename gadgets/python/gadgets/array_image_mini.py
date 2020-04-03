from __future__ import print_function
import numpy as np
from gadgetron import Gadget

class ArrayImage(Gadget):
    def __init__(self,next_gadget=None):
        super(ArrayImage,self).__init__(next_gadget=next_gadget)
        

    def process_config(self, cfg):
        pass
       
    def process(self, header, image, metadata=None):


            print(image.shape)

      
            #Send the combined image and the modified header and metadata
            if metadata is not None:
                self.put_next(header,image,metadata)
            else:
                self.put_next(header,image)
         
        return 0

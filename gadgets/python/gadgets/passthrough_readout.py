import sys
import ismrmrd
import ismrmrd.xsd
import numpy as np
from gadgetron import Gadget

class ReadoutPassThrough(Gadget):
    def __init__(self,next_gadget=None):
        super(ReadoutPassThrough,self).__init__(next_gadget=next_gadget)
        self.counter = 0
        self.counter_send = 0

    def process_config(self, conf):
        pass 

    def process(self, header, data):
         #Return data to Gadgetron
        self.counter=self.counter+1;

        slice = header.idx.slice
        repetition=  header.idx.repetition
        e1=header.idx.kspace_encode_step_1
        segment=  header.idx.segment

        if (repetition==0 and slice==0):
          #print(self.counter, slice , repetition)
          print(np.shape(data))
          #print(header.scan_counter)
          self.counter_send =self.counter_send+1
          print(self.counter_send, " slice: ",slice , " rep: ", repetition, " e1: ", e1," segment: ",  segment)
          self.put_next(header,data)

        return 0    
        #print "Returning to Gadgetron"
 

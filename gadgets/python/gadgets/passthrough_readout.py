import sys
import ismrmrd
import ismrmrd.xsd


from gadgetron import Gadget

class ReadoutPassThrough(Gadget):
    def __init__(self,next_gadget=None):
        super(ReadoutPassThrough,self).__init__(next_gadget=next_gadget)
        self.counter = 0

    def process_config(self, conf):
        pass 

    def process(self, header, data):
         #Return data to Gadgetron
        self.counter=self.counter+1;
        print(self.counter)
        self.put_next(header,data)
        return 0    
        #print "Returning to Gadgetron"
 

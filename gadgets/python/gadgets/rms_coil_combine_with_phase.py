import numpy as np
from gadgetron import Gadget
import nibabel as nib
try:
 import matplotlib.pyplot as plt
except:
  print("error  import matplotlib.pyplot as plt")

 
def create_complex(rho, theta):             

        if (np.all(np.equal(rho.shape,theta.shape))):
          output = np.zeros(np.shape(rho), dtype=np.complex)
          real_part=rho * np.cos(theta)
          imag_part=rho * np.sin(theta)
          output=real_part +imag_part *1j 
        else: 
          output=rho
          print("RMSCoilCombineWithPhase: issue with size of rho, and theta, size does not match")

        return output

class RMSCoilCombineWithPhase(Gadget):
    def __init__(self, next_gadget=None):
        Gadget.__init__(self,next_gadget)
        self.compteur=[]
        pass
    

    def process_config(self, cfg):
        print("RMS Coil Combine, Config ignored")
        self.compteur=0

    def process(self, h, im):
        last_dims=len(im.shape)-1
        
        #combined_image = np.sqrt(np.sum(np.square(np.abs(im)),axis=last_dims))
        mag= np.sqrt(np.sum(np.square(np.abs(im)),axis=last_dims))         

        mag_tmp=np.square(np.abs(im))        
        phase_tmp=np.angle(im)  
        tempo=mag_tmp*phase_tmp   
     
        phase=np.sum(tempo,axis=last_dims)   
        output=create_complex(mag, phase) 
        
        if (h.repetition==0 and self.compteur ==0):
           print( np.amin(mag) ,  np.amax(mag)) 
           print( np.amin(phase) ,  np.amax(phase)) 
           #np.save('/tmp/gadgetron/magnitude', mag)
           #np.save('/tmp/gadgetron/phase', phase)
           #plt.figure(1)
           #print(np.shape(mag))
           #plt.imshow(mag[:,:,0])
           #plt.show()
           #plt.pause(5) 
           #export_phase=phase-np.amin(phase)
           #print( np.amin(export_phase) ,  np.amax(export_phase)) 
           #ni_img = nib.Nifti1Image(mag)
           #nib.save(ni_img, '/tmp/gadgetron/magnitude.nii.gz')
           #ni_img = nib.Nifti1Image(phase)
           #nib.save(ni_img, '/tmp/gadgetron/phase.nii.gz')
           #print("nib ok")
           #plt.close()  
                     
        self.compteur=self.compteur+1
        print("RMS coil",im.shape,output.shape)
        h.channels = 1
        self.put_next(h,output.astype('complex64'))
        return 0



import numpy as np
import dwilib.dwi.models

model = model = [x for x in dwilib.dwi.models.Models if x.name == 'StretchedN'][0]
images = 100*np.ones((10,10,3))
images[1,0] = 0.8
images[1,1] = 0.7
bvals = [0, 20, 500]
bvals = np.asanyarray(bvals)
images = images.reshape(-1,len(bvals))
pmap = model.fit(bvals,images)
1+1
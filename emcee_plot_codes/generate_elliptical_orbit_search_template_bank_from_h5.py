import emcee
import numpy as np
sampler = emcee.backends.HDFBackend("elliptical_template_bank_sample.h5")
templates_required = 100 # test (change this based on number of templates required)
flatchain = sampler.get_chain(flat=True)
indices = np.random.choice(flatchain.shape[0], templates_required, replace=False)
templates_for_search = flatchain[indices]

with open('5D_template_bank_sample', 'w') as f:
    
    for i in range(len(templates_for_search)):
        
    
        f.write(str(templates_for_search[i][0]) + ' ' + str(templates_for_search[i][1]) + ' ' + \
                str(templates_for_search[i][2]) + ' ' + str(templates_for_search[i][3]) + ' ' + \
                str(templates_for_search[i][4]) + ' ' + str(2 * np.pi/templates_for_search[i][0]) + ' ' + \
                str(templates_for_search[i][2]/(2 * np.pi)) + '\n')

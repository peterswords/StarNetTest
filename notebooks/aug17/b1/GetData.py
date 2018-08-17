import numpy as np
import h5py

DATADIR="/media/apogee/starnet/aug17/"

def getDataDir():
    return DATADIR
    
def getParams():
    # The list of parameters we will train and test on
    return ['TEFF', 'FE_H', 'ALPHA_M', 'C_FE', 'N_FE' ]

def getCols():
    # Additional columns needed for test/training set selection
    
    # For the testing of StarNet, it is necessary to obtain the spectra, error spectra, combined S/N, and labels,
    # but we need to make eliminations to the test set to obtain the labels of highest validity to compare with,
    # so we will first include the APOGEE_IDs, the S/N of the combined spectra, $T_{\mathrm{eff}}$, $\log(g)$,
    # [Fe/H], $V_{scatter}$, STARFLAGs, and ASPCAPFLAGs to make certain eliminations.    
    
    cols =  getParams().copy()
    cols.extend(['stacked_snr', 'LOGG', 'star_flag', 'aspcap_flag', 'VSCATTER', ])
    print(cols)
    return cols
    
def get( F, only_high_snr ):
    print('Dataset keys in file: \n')
    print(list(F.keys()))
    
    data = {'IDs': F['IDs'][:,0]}    
    for col in getCols():
        data[col] = F[col][:]
    
    print('Obtained data for '+str(len(list(set(list(data['IDs'])))))+' stars.')

    teff_min, teff_max, vscatter_max, snr_min, metal_min, metal_max = 4000., 5500., 1., 200., -3., 10.
     
    flags = ((data['aspcap_flag'][:]==0.) & (data['star_flag'][:]==0.) &
        (data['VSCATTER'][:] < vscatter_max) & (data['LOGG'][:]!=-9999.) &
        (data['TEFF'][:] > teff_min) & (data['TEFF'][:] < teff_max))
        
    print('main flags '+str(sum(flags)))
    
    if only_high_snr:
        flags = flags & (data['stacked_snr'][:]>=snr_min)
        print('snr flags '+str(sum(flags)))

    for metal in ['FE_H', 'ALPHA_M', 'C_FE', 'N_FE']:
        flags = flags & (data[metal][:] > metal_min) & (data[metal][:] < metal_max)    
        print(metal +' flags '+str(sum(flags)))

    indices, cols = np.where((flags).reshape(len(data['IDs']),1))

    data['indices'] = indices
    return data
    
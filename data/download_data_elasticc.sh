#!/bin/bash

classes=(Cepheid EB RRL SNII-Templates SNIa-SALT3 SNIb-Templates SNIc-Templates TDE d-Sct)
pmin=(1 60 0.3 None None None None None 0.01)     #minimum period [d]
pmax=(100 80 1.2 None None None None None 0.2)    #maximum period [d]
objidx=(19 1419 357 None None None None None)    #idx of objectr to use
echo ${classes[@]}

# for ((i=0; i<${#classes[@]}; i++)); do
for ((i=3; i<4; i++)); do
    echo "${classes[i]} ${pmin[i]} ${pmax[i]} ${objidx[i]}"

    wget -O ${classes[i]}.FITS.gz https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/ELASTICC2_TRAIN_02_${classes[i]}/ELASTICC2_TRAIN_02_NONIaMODEL0-0001_PHOT.FITS.gz
    python3 get_data_elasticc.py ${classes[i]}.FITS.gz ${pmin[i]} ${pmax[i]} ${objidx[i]}

    rm ./${classes[i]}.FITS.gz
    
done
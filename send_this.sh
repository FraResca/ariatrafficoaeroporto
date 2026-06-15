#!/bin/bash

cd ..

rsync -avz --progress --exclude '.conda-env' /home/fresca/repos/ariatrafficoaeroporto/ fresca@copernico.unife.it:/hpc/home/fresca/ariatrafficoaeroporto

# rsync -avz --progress --exclude '.conda-env' fresca@copernico.unife.it:/hpc/home/fresca/ariatrafficoaeroporto/ /home/fresca/repos/ariatrafficoaeroporto/

# conda env update -p ./.conda-env -f environment.yml

# scp -r ariatrafficoaeroporto fresca@copernico.unife.it:/hpc/home/fresca/ariatrafficoaeroporto

# get this back
# scp -r fresca@copernico.endif.man:/hpc/home/fresca/ariatrafficoaeroporto .


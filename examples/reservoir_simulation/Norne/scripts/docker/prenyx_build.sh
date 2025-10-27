#!/bin/bash
#############################################################
# Author: Clement Etienam (cetienam@nvidia.com)
#############################################################


srun \
    -p b200-a01r \
    -N 1 \
    -A coreai_devtech_all -J coreai_devtech_all-total_rft_2025:dev \
    --ntasks-per-node=8 \
    --comment="Build GEOSX stack" \
    -t 04:00:00 \
    --container-image="gitlab-master.nvidia.com/globalenergyteam/customers/total/total_rfp_reservoir:athena_x86" \
    --container-mounts=/lustre/fsw/coreai_devtech_all/cetienam/physicsnemo_norne:/workspace/project \
    --container-env="OMPI_MCA_plm=slurm" \
    --container-env="OMPI_MCA_btl=self,tcp,vader" \
    --container-env="OMPI_MCA_pmix=^s1,s2,cray" \
    --container-env="PMIX_MCA_gds=hash" \
    --container-workdir=/workspace/project \
    --pty /bin/bash

# pgrad-thesis
## Impromptu List of Things to Re-create Environment
conda create -n pgrad-thesis -c conda-forge -c rapidsai-nightly python=3.8 cudatoolkit=11.2 cucim tensorflow-gpu pandas matplotlib

(for rasterio)
conda config --add channels conda-forge
conda config --set channel_priority strict

conda install -y rasterio tqdm

# Updates
## Jan-10
- Original data (Maxar) is blurry and unlabelled, infeasible for MSc without labelling/ground 


# Experiment Managment/Tracking
## General Notes
- Dice coefficient (and focal loss) used to evaluate performance, NOT accuracy


# Yes, I know there are better ways to do this
- [x] Get a single training loop done with BW images
- [x] Add validation step to training loop
- [x] Add SEGNET and RGB data pipeline
- [ ] Fix loss function/model output, error where cannot calculate Dice loss
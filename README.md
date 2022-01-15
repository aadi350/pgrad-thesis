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
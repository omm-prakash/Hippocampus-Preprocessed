# setup environment
conda create --name daad-task
conda activate daad-task

# installing pacakages and libraries
conda install -c anaconda pandas numpy matplotlib scikit-learn
conda install -c conda-forge nibabel keras tensorflow

# installing code editor
conda install jupyter notebook

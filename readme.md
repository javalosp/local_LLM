# Running on Imperial College HPC:

The standard virtual environment (venv) only selfcontains python packages
For packages that require compiling C, C++,... code we need to have the appropriate build tools
Therefore it's necessary to use a venv that also self-contains those tools
-> Create a Conda venv

From a Terminal
1. Load Conda

eval "$(~/miniforge3/bin/conda shell.bash hook)"

# Create a new environment named 'llm_env' with Python and the required build tools
conda create --name llm_venv python=3.9 gxx_linux-64 cmake -c conda-forge -y

# Activate the new environment
conda activate llm_venv

# Update pip
pip install --upgrade pip

# For GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir llama-cpp-python

# Install required libraries
pip install -r requirements.txt

Now you have a virtual environment with the required packages.

For running on a Jupyterlab notebook

It is necessary to install a kernel based on the virtual environment

# Install jupyterlab kernel

From a Terminal:

python -m ipykernel install --user --name llm_venv --display-name "llm_venv (kernel)"

After this the "llm_venv (kernel)" should be available in the list of kernels for the notebook

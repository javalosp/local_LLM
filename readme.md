# Local LLM Pipeline

This project provides a local-only Retrieval-Augmented Generation (RAG) pipeline. It uses open-source Large Language Models (LLMs) to answer questions based on a set of local documents. The entire process, from document processing to model inference, runs on your machine without requiring any API keys or internet access after the initial model download.

### Features

* **100% Local**: All components, including the LLM, run locally. No data is sent to external services.
* **Document-Based Q&A**: Review PDF documents from a local folder and use them as a knowledge base.
* **Swappable LLMs**: Easily switch between different open-source LLMs like Mistral, Llama 3 (Meta), Phi-3 (Microsoft), and Gemma (Google).
* **Efficient Vector Storage**: Documents are only processed once by creating a local vector database with the documents information.
* **Registration-Free Models**: The model catalog is pre-configured to use non-gated GGUF model repositories, allowing for easy downloads.
* **Dynamic Queries**: Ask questions directly (hardcoded) or provide them via a text file. A user interface can be integrated in the future.

### Project Structure.
```bash
    ├── sources/              # Contains the source PDF documents
    ├── models/               # LLM models will be downloaded here automatically
    ├── vectorstore/          # Stores FAISS vector databases
    ├── main.py               # Main execution script
    ├── preprocess.py         # Handles document loading and vectorisation
    ├── llm_model_setup.py    # Manages LLM model downloading and setup
    ├── catalogs.py           # Contains the catalog of available LLMs
    ├── settings.py           # Main configuration file (Set up your model settings here)
    ├── requirements.txt      # Python dependencies
    └── llm_notebook.ipynb    # Jupyter notebook for interactive testing
```

## Getting Started
Follow these steps to set up and run the project on your local machine.
1. Prerequisites: 
    * Python 3.9 or higher.
        * Required libraries (found in `requirements.txt`):
            ```bash
            jupyterlab
            langchain
            langchain-community
            pypdf
            sentence-transformers
            faiss-cpu
            llama-cpp-python
            ```
        It is recommended to set up a virtual environment to manage dependencies.

    * Git for cloning the repository.
2. Clone the Repository:
    ```bash
    git clone https://github.com/javalosp/local_LLM
    cd local_LLM
    ```
3. Set Up a Virtual Environment:
    ```bash
    # Create a virtual environment
    python -m venv /path/to/new/virtual/environment
    # Activate it
    # On Windows:
    C:\path\to\new\virtual\environment\Scripts\activate
    # On macOS/Linux:
    source /path/to/new/virtual/environment/bin/activate
    ```
4. Install Dependencies:
    Install all the required Python libraries using the requirements.txt file.
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. Add Your Documents: 
Place all the PDF files you want the LLM to review into the `sources/` directory.

2. Select Your LLM:
Open the settings.py file and choose the model you want to use.
The MODELS_CATALOG in catalogs.py lists all available options.

The first time you run the script with a new model, it will be automatically downloaded from Hugging Face and saved to the `models/` directory. This may take some time depending on the model size and your internet connection.

## How to Run

#### You can run the pipeline from your terminal.
1. Basic Execution
To run with the default, hardcoded query in main.py:
    ```bash
    python main.py
    ```
2. Using a Query from a File
You can provide a query from a text file using the `--query_file` argument.
* Create a file, e.g., [question.txt](./question.txt), and write your question in it
* Run the script using the `--query_file` flag:
    ```bash
    python main.py --query_file question.txt
    ```

The script will then load the documents, create or load the vector store, initialise the LLM, and generate an answer based on your query and the provided context.

#### Interactive Notebook
For experimentation and debugging, you can use the [llm_notebook.ipynb](llm_notebook.ipynb) jupyter notebook. It contains the same pipeline logic broken down into cells, allowing you to inspect each step of the process.
To use it, start a Jupyter Lab session, then open the [llm_notebook.ipynb](llm_notebook.ipynb) file in the Jupyter interface.


# Running on Imperial College HPC:

It is recommended to use [Conda](https://docs.conda.io/en/latest/index.html) as explained in the [Imperial College London's RCS User Guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/conda/):

1. Install and set up **conda-forge** (*This is required only the first time*)
    ```bash
    module load miniforge/3
    miniforge-setup
    ```
2. Load Conda
    ```bash
    eval "$(~/miniforge3/bin/conda shell.bash hook)"
    ```
3. Create a new environment named with Python and the required build tools for the dependencies
    ```bash
    conda create --name venv_name python=3.9 gxx_linux-64 cmake -c conda-forge -y
    ```
4. Activate the new environment
    ```bash
    conda activate venv_name
    ```
5. Install required libraries    
    ```bash
    # Update pip
    pip install --upgrade pip

    # Install required libraries
    pip install -r requirements.txt

    # For GPU support (llama-cpp-python)
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir llama-cpp-python
    ```

#### Using the jypyter notebook on the HPC
For using the jupyter notebook on the HPC it is necessary to install a kernel based on the virtual environment

1. Install jupyterlab kernel

From a Terminal run:
```bash
python -m ipykernel install --user --name venv_name --display-name "venv_kernel_name"
```

2. Start a JupyterLab session
    https://jupyter.cx3.rcs.ic.ac.uk/

Now the kernel `venv_kernel_name` should be available in the list of kernels for the notebook


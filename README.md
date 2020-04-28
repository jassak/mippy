# mippy
Medical Informatics Platform in python

## Getting started

1. Install `python3.8` [here](https://www.python.org/downloads/)
1. `git clone https://github.com/jassak/mippy.git`
2. `cd mippy`
3. Create virtual environment
    ```bash
    python3.8 -m venv mip-venv
    ```
4. Activate virtual env 
    ```bash
    source mip-venv/bin/activate
    ```
5. Install requirements
    ```bash
   pip install -r requirements.txt 
   ```
6. Install mippy in editable state (all the edits made to the .py files will be 
    automatically included in the installed package)
    ```bash
    pip install -e .
   ```
7. Run pyro4 name server
    ```bash
   pyro4-ns 
   ```
8. To run a machine learning algorithm we first need to initialize the local servers. 
    Currently there are three by default.
    ```bash
    python mippy/localnode.py
    ```
9. In a separate shell activate venv again (see step 4.). Then run some algorithm like this
    ```bash
    python mippy/ml/*some_algorithm.py* 
    ```
    where you have to replace `*some_algorithm.py*` by the script you want to run.
    With no arguments, the algorithm runs with some default parameters. If you want to 
    run it with your parameters pass them as arguments. To see the available arguments
    run
    ```bash
    python mippy/ml/*some_algorithm.py* --help
    ```
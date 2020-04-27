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
6. Run pyro4 name server
    ```bash
   pyro4-ns 
   ```
7. Run Logistic Regression on 3 local DBs:
    - Instantiate servers on three local nodes
        ```bash
       python src/localnode.py
       ```
    - Run logistic client in separate shell (you need to activate venv again)
        ```bash
        python src/logistic_regression.py
       ```
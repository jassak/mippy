# mippy
Medical Informatics Platform in python

## Getting started

1. Install `python3.8` [here](https://www.python.org/downloads/)
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
7. Run logistic server
    ```bash
   python3 src/logistic_server.py 
   ```
7. Run logistic client in separate shell (you need to activate venv again)
    ```bash
    python3 src/logistic_client.py
   ```
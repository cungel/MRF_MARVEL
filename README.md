# MRF MARVEL

This repository accompanies the work presented in *ISMRM 2025* and was developed by the **MR-Vascular Fingerprinting team** at the *Grenoble Institute of Neurosciences (GIN)*.

---

## Requirements

The code is compatible with **Python 3.12**. All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```
## Project Structure

- `MARVEL/ & MARVEL_complex/`  
  These two directories are built on the same basis. They each contain:
    - a train.py script used during training,
    - a reconstruction.py script used to apply the network on new data.

- `Matching/`  
  Implements traditional dictionary matching approaches:
  - Matching on magnitude-only signals
  - Matching on complex signals (magnitude + phase)

- `Neural_Networks/`  
  Contains the architecture definitions for both versions of MARVEL.

- `Plot/` and `Tool/`  
  - `Plot/`: Visualization utilities for generating parameter maps and figures.
  - `Tool/`: Utility functions for dictionary generation, signal processing, and model training.

- `Undersampling/`  
    This is the main directory for the retrospective and prospective studies.
    It includes:
    - `Retrospective_study/Examples/`: Commented scripts to simulate numerical phantoms and evaluate performance.
    - `Prospective_study/`: Scripts for processing in-vivo data.
    - `CG-SENSE/`: Adapted CG-SENSE reconstruction from TensorFlow-MRI, compatible with current GPU environments.

- `requirements.txt`  
  Lists all dependencies required to run the codebase.

## References
* 1. Barrier, A., Coudert, T., Delphin, A., Lemasson, B. & Christen, T. MARVEL: MR Fingerprinting with Additional micRoVascular Estimates Using Bidirectional LSTMs. in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024 (eds. Linguraru, M. G. et al.) vol. 15002 259–269 (Springer Nature Switzerland, Cham, 2024).

* 2. (ISMRM 2022) TensorFlow MRI: A Library for Modern Computational MRI on Heterogenous Systems. https://archive.ismrm.org/2022/2769.html.

* 3. Ong, F., & Lustig, M. (2019). SigPy: A Python Package for High Performance Iterative Reconstruction. Proceedings of the 27th Annual Meeting of ISMRM, 4819.

## Contacts
For questions, please contact: lila.cunge@gmail.com
Institut des Neurosciences de Grenoble – MR Fingerprinting team
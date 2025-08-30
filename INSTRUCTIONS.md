
# Instructions for Dataset Preparation and Pre-training

This document provides instructions for preparing the dataset and running the pre-training for the HRM-rnn-base model.

## 1. Dataset Preparation

The dataset preparation script `dataset/build_arc_dataset.py` processes the raw ARC dataset and creates a version with augmentations.

**To run the dataset preparation:**

1.  **Navigate to the project's root directory:**
    ```bash
    cd /home/alpha/research/HRM-rnn-base
    ```

2.  **Run the `build_arc_dataset.py` script:**
    ```bash
    python3 dataset/build_arc_dataset.py
    ```
    This will process the datasets specified in `dataset_dirs` within the `DataProcessConfig` class in the script and save the processed data to the `output_dir` (by default, `data/arc-aug-1000`).

## 2. Pre-training

The `pretrain.py` script is used to pre-train the model.

**To run the pre-training:**

1.  **Navigate to the project's root directory:**
    ```bash
    cd /home/alpha/research/HRM-rnn-base
    ```

2.  **Install the required dependencies:**
    The `hrm_v3` model requires a specific version of the `sru` library. Install it using the following command:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the `pretrain.py` script:**

    *   **To run the `v2` variant (SRU):**
        ```bash
        python3 pretrain.py --arch=hrm_v2
        ```

    *   **To run the `v3` variant (SRU++):**
        ```bash
        python3 pretrain.py --arch=hrm_v3
        ```

    You can also modify other configuration parameters by passing them as command-line arguments.

    **Example of overriding a configuration parameter:**
    ```bash
    python3 pretrain.py --arch=hrm_v3 --arch.hidden_size=256
    ```

    For more information on the available command-line arguments, run:
    ```bash
    python3 pretrain.py --help
    ```

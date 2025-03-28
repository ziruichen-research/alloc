## Analogical Learning-Based Wireless Localization (ALLoc)

Zirui Chen, Zhaoyang Zhang, Ziqing Xing, Ridong Li and Zhaohui Yang

------

This repository is the official implementation of paper `Analogical Learning for Cross-Scenario Generalization: Framework and Application to Wireless Localization`. All experimental results presented in this paper can be reproduced using this project. 

### Usages

#### Environment requirements

To ensure dataset partitioning and expansion, approximately 1TB of storage space is required. We train the Mateformer on the NVIDIA Hopper architecture, requiring around 50GB of GPU memory (both the model and data are based on float32). If the device has insufficient memory, the `batch_size` parameter in the code can be appropriately reduced (default `batch_size=500`).
For detailed library dependencies, please refer to `requirements.txt`.

#### Dataset

This work is based on the [DeepMIMO dataset](https://arxiv.org/abs/1902.06435) (V1), filtering out samples with zero or excessively small channel amplitudes, as such samples cannot be collected in real-world systems. To facilitate deep learning, we restructured the data format and implemented a fast data loading interface in the code. The restructured dataset can be downloaded via the following links. 

*Google Driven*: https://drive.google.com/drive/folders/1xSo7-liilx3idr70h8hBdGOeegGFXRHO?usp=drive_link

*Baidu Netdisk*: https://pan.baidu.com/s/11v5ZAaIlNzta3aD7axZpbQ?pwd=f7a4

After downloading the data, please place these compressed files in the `ALLoc` folder (as a `/Data` subfolder), and run `bash decompress.sh` inside the `/Data` subfolder to decompress them.

#### Training and Testing

The overall file structure of the repository is as shown in the `directory_structure.txt` file. 

- The `/O1` and `/O1B` folders correspond to the **Single-Scenario Learning and Generalization** section in the text.
- The `/transfer_O1model_to_O1Btest` and `/transfer_O1model_to_O1Btest` folders correspond to the **Cross-Scenario Learning and Generalization** section (thus requiring the models in `/O1` and `/O1B` to be trained first).
- The `/MO1` folder corresponds to the **Multi-Scenario Learning and Generalization** section.

Within each scenario, the project is implemented in a parallel folder structure. For example, the main differences between `/train10000` and `/train20000` are the amount of training data, while the rest of the code remains nearly consistent.

Before running the code, it is needed to replace the file and data paths from our hardcoded paths to the new current directory. You can do this by running `bash pathchange.sh` using the main folder as the working directory. We also provide several `.sh` script programs to help you quickly run experiments and better understand the relationships between these files.

To complete data generation, training, and testing of ALLoc in O1 scenario (with 40,000 training data), simply run `bash single_scenario_O1.sh`. 

In `single_scenario_O1.sh`:

- `datadivision.py` is used to randomly choice a subset from the original channel-location pairs of DeepMIMO as the training and testing datasets for the localization task.
- `datadivision_sequence.py` is used to construct the neighborhood sampling sequence for each sample based on the channel-location pairs in the training set.
- `datadivision_test_error_len64.py` is used to search the training set for neighborhood sampling sequences required for inference on the test set samples, leveraging coarse location information.
- `train.py` and `test.py` are responsible for training and testing ALLoc, respectively.

(We provide all model files, so testing can be performed directly without training. If you need to reproduce the training process, simply uncomment the line `python train.py` in the `single_scenario_O1.sh` script.)

To complete data generation, training, and testing of ALLoc in O1B scenario (with 40,000 training data points), simply run `bash single_scenario_O1B.sh`. This script is nearly identical to `single_scenario_O1.sh`, with only changes in the scenario parameters.

After running `single_scenario_O1.sh`, you can proceed with cross-scenario performance testing. Navigate to the `/transfer_O1model_to_O1Btest/train40000/Proposed/Scheme/direct_test` directory and run `python test.py`.

To complete data generation, training, and testing of ALLoc in MO1 scenario, simply run `bash multi_scenario.sh`. The code logic in this script is nearly consistent with `single_scenario_O1.sh`, except that the specific `.py` files include data from five scenarios.

For other parts of training and testing, you can infer the process from the file names. When running these scripts, please use the respective subdirectory as the working directory instead of the entire `ALLoc` directory.

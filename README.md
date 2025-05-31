# 🫁 LUNA25 Challenge

[LUNA25 Challenge](https://luna25.grand-challenge.org/) uses artificial intelligence for **lung nodule malignancy risk estimation** on low-dose chest CT scans. 

Accurate malignancy risk estimation of pulmonary nodules is critical for early lung cancer detection and intervention. In this study, we compare multiple 2D and 3D deep learning models on Luna25 dataset using a strict patient-level split to classify pulmonary nodules from low-dose chest CT scans. Among 2D CNNs, Resnet18 (baseline), ResNet50, EfficientNet-B0 achieved top AUC scores close to $0.88$, with a stacked ensemble improving it to $0.89$. The 3D ResNet3D MC3 model, leveraging mixed spatio-temporal convolutions and transfer learning, matched this performance. A **3D Vision Transformer** outperformed all other models, showing the highest AUC of **$0.90$**. These results highlight the benefits of architectural diversity in volumetric medical data 

## 🗂️ Content
This algorithm provides a framework for training and testing models.

Important Files:
- `train.py`: A script for training the algorithm on local data.
- `inference.py`: A script for testing the trained algorithm using a specified configuration.
- `Dockerfile`: A file to build a Docker container for deployment on Grand-Challenge. For help on setting up Docker with GPU support you can check the documentation on [Grand-Challenge](https://grand-challenge.org/documentation/setting-up-wsl-with-gpu-support-for-windows-11/) or [Docker](https://docs.docker.com/engine/install/ubuntu/) for additional information.

## 🛠️ Setting up the Environment
To set up the required environment for the baseline algorithm:
1. **Create an environment and esure Python is Installed**: Install Python 3.9 or higher:
    ```bash
    conda create -n luna25-baseline python==3.9
    ```
2. **Install Dependencies**:
    - Run the following command to install the dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3. **Verify Installation**:
    - Test the installation by running:
    ```bash
    python --version
    pip list
    ```
    Ensure all required packages are listed and no errors are reported.

## 🏋️ Performing a Training Run
1. **Set up training configurations**

Open `experiment_config.py` to edit your training configurations. Key parameters include:

- `self.MODE`: Set this to 2D or 3D depending on the desired baseline model.
- `self.EXPERIMENT_NAME`: Specify the name of your experiment (e.g. LUNA25-baseline).
- `self.CSV_DIR_TRAIN`: the path to the training csv file
- `self.DATADIR`: the path where the images are stored

More about configuration parameters can be found below.

2. **Training the Model**

To train the model using the `train.py` script:
```bash
python train.py
```
This script uses the settings from experiment_config.py to initialize and train the model.

## 🧪 Testing the Trained Algorithm
1. **Configure the inference script**

Open the `inference.py` script and configure:
- `INPUT_PATH`: Path to the input data (CT, nodule locations and clinical information). Keep as `Path("/input")` for Grand-Challenge.
- `RESOUCE_PATH`: Path to resources (e.g., pretrained models weights) in the container. Defaults to `/results` directory (see Dockerfile)
- `OUTPUT_PATH`: Path to store the output in your local directory. Keep as `Path("/output")` for Grand-Challenge.
- **Inputs for the `run()` function**:
    - `mode`: Match this to the mode used during training (2D or 3D).
    - `model_name`: Specify the experiment_name matching the training configuration (corresponding to experiment_name directory that contains the model weights in `/results`).

2. **Updating the Docker Image Tag**

In `do_test_run.sh`, update the Docker image tag as needed:
```bash
DOCKER_IMAGE_TAG="luna25-baseline-3d-algorithm-open-development-phase"
```


3. **Running the Test Script**

To test the trained model for running inference run: 
```bash
./do_test_run.sh
``` 

This script performs the following:
- Uses Docker to execute the `inference.py` script.
- Mounts necessary input and output directories.
- Adjusts the Docker image tag (if updated) before running.

## 🐳 Building the Docker Image
To build the Docker container required for submission to Grand-Challenge run:
```bash
./do_save.sh
```
This will output a *.tar.gz file, which can be uploaded to Grand-Challenge.
More information on testing and deploying your container can be found [here](https://grand-challenge.org/documentation/test-and-deploy-your-container/).


## ⚙️ Configuration Settings

Below are the key configuration parameters used in the project:

### Directories
- **`WORKDIR`**: `./` – Base working directory.
- **`RESOURCES`**: `./resources` – Directory containing resource files such as model weights.
- **`MODEL_RGB_I3D`**: `./resources/model_rgb.pth` – Path to the pretrained I3D model weights.
- **`DATADIR`**:  Directory containing LUNA25 nodule blocks.
- **`CSV_DIR`**: `./datasets` – Folder containing CSV files for training and validation.
- **`CSV_DIR_TRAIN`**: `./datasets/train.csv` – Training data CSV.
- **`CSV_DIR_VALID`**: `./datasets/valid.csv` – Validation data CSV (can be `None`).
- **`FOLDS`**: `None` – Set to `None` if not training an ensemble on different folds, otherwise input the number of folds wanted.
- **`EXPERIMENT_DIR`**: `./results` – Directory where experiment results will be saved.
- **`EXPERIMENT_NAME`**: `"LUNA25-ViT"` – Identifier for the current experiment.
- **`MODE`**: `"3D"` – Set to `"2D"` or `"3D"` depending on the model mode.
- **`MODEL`**: `"vit"` – Model architecture to use.

### Training Parameters
- **`SEED`**: `2025` – Random seed for reproducibility.
- **`NUM_WORKERS`**: `8` – Number of data loading workers.
- **`SIZE_MM`**: `50` – Physical size in mm for each patch.
- **`SIZE_PX`**: `64` – Patch size in pixels.
- **`BATCH_SIZE`**: `32` – Batch size for training.
- **`ROTATION`**: `((-45, 45), (-45, 45), (-45, 45))` – 3D rotation range for data augmentation.
- **`TRANSLATION`**: `True` – Enable translation augmentation.
- **`EPOCHS`**: `100` – Number of training epochs.
- **`PATIENCE`**: `20` – Early stopping patience.
- **`PATCH_SIZE`**: `[64, 128, 128]` – Shape of 3D patches (Depth, Height, Width).
- **`LEARNING_RATE`**: `1e-4` – Learning rate.
- **`WEIGHT_DECAY`**: `5e-4` – L2 regularization term.

### Loss Function
- **`LOSS`**: `"BCE"` – Loss function type (Binary Cross Entropy).
- **`POS_WEIGHT`**: `10.0` – Weight for the positive class in BCE loss.

### Vision Transformer (ViT) Parameters

```python
VIT = {
    "image_size": (64, 64),     # Spatial dimensions (height, width)
    "frames": 64,                 # Temporal depth (number of slices/frames)
    "image_patch_size": 16,      
    "frame_patch_size": 8,       
    "dim": 1024,                  # Embedding dimension
    "depth": 6,                   # Number of transformer blocks
    "heads": 8,                   # Number of attention heads
    "mlp_dim": 1024,              # MLP layer dimension
    "dropout": 0.1,               # Dropout rate
    "emb_dropout": 0.1            # Embedding dropout rate
}
```
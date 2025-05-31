from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("./")
        
        self.RESOURCES = Path("./resources")
        
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (self.RESOURCES / "model_rgb.pth")
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group07/luna25_nodule_blocks/")
        

        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("./datasets")

        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV
        self.FOLDS = None  # Set to None if you are not using cross-validation

        # To train the ensemble just set validation set to None and use FOLDS
        #self.CSV_DIR_TRAIN = self.CSV_DIR / "LUNA25_Public_Training_Development_Data.csv" # Path to the training CSV
        #self.CSV_DIR_VALID = None
        #self.FOLDS = 10

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "LUNA25-ViT"
        self.MODE = "3D" # 2D or 3D
        self.MODEL = "vit"

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 8
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = ((-45, 45), (-45, 45), (-45, 45))
        self.TRANSLATION = True
        self.EPOCHS = 100
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 5e-4

        self.LOSS = "BCE"
        self.POS_WEIGHT = 10.0

        # Model parameters
        self.VIT = {
            "image_size": (self.SIZE_PX, self.SIZE_PX), # image size
            "frames": self.SIZE_PX, # number of frames
            "image_patch_size": 16,     # image patch size
            "frame_patch_size": 8,      # frame patch size
            "dim": 1024,  
            "depth": 6,
            "heads": 8,
            "mlp_dim": 1024,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }

    def __repr__(self):
        return f"Configuration({self.__dict__})"

config = Configuration()
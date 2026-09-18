"""Paths and constants shared by the pipeline (see CLAUDE.md for the model contract)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "flat-connectome"
CACHE_DIR = ROOT / "data" / "cache"

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_TRACED = "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
SYN_PARTNERS_TRACED = "syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather"

# consensus_nt values are full lowercase names; dopamine/octopamine/serotonin/unclear → 0
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}

SOURCE_TYPES = ("LPLC2", "LC4")
MOTOR_SUPERCLASSES = ("vnc_motor", "cb_motor")
MIN_WEIGHT = 5
MAX_HOPS = 3
NODE_CAP = 4000
MN_CAP = 300
EDGE_CAP = 150_000
DN_CAP = 300  # descending neurons kept, by forward flow from the sources (Giant Fiber reserved)

VOXEL_NM = 8  # annotation + synapse coordinates are 8 nm voxels; shell meshes are nm

# Verified in tests/test_real_positions.py: the fly's LEFT side has LARGER x (image orientation)
LEFT_IS_POSITIVE_X = True

# --- simulation contract (torch, numpy and web/model.js; written into the bundle manifest) ---
FPS = 60
SUBSTEPS = 4
DT_SUB = 1.0 / (FPS * SUBSTEPS)  # seconds
R_MAX = 4.0
WARMUP_FRAMES = 60
CLASSES = ("none", "dodge_right", "dodge_left", "takeoff")
DECISION_P = 0.7
DECISION_FRAMES = 3

# --- looming stimulus (azimuth degrees: -90 left, 0 front, +90 right) ---
INPUT_KINDS = ("LPLC2", "LC4")
RF_CENTERS_DEG = (-45.0, 45.0)  # eye of side 1 (L), eye of side 2 (R)
RF_SIGMA_DEG = 50.0
THETA_SAT_DEG = 90.0
THETADOT_SAT_DPS = 1000.0
LABEL_ONSET_DEG = 15.0
FRONT_HALF_WIDTH_DEG = 45.0

# --- model initialisation ---
TAU_MIN_MS, TAU_MAX_MS, TAU_INIT_MS = 5.0, 200.0, 20.0

# --- bundle / world coordinates ---
WORLD_AXES = ((-1, 0, 0), (0, 0, -1), (0, -1, 0))  # world = M·(nm - centre)/1000; det +1
VIS_EDGES = 50_000
DUST_SCALE_UM = 0.025

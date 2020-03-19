BALANCED_CLASS_DISTRIBUTION = {
    'INDUSTRIAL': 0.25,
    'PUBLIC': 0.16,
    'RETAIL': 0.11,
    'OFFICE': 0.1,
    'OTHER': 0.07,
    'AGRICULTURE': 0.02,
    'RESIDENTIAL': 0.29
}

BALANCED = 'balanced'
IMBALANCED = 'imbalanced'
ALL = 'all'

DATA_SET_TYPES = [BALANCED, IMBALANCED, ALL]

MODELATE_RAW_DATA_SET = 'Modelar_UH2020.txt'
ESTIMATE_RAW_DATA_SET = 'Estimar_UH2020.txt'

TRAIN = 'train'
TEST = 'test'

DATA_FILE_NAMES_BALANCED = {
    TRAIN: '{}_data_{}.csv'.format(TRAIN, BALANCED),
    TEST: '{}_data_{}.csv'.format(TEST, BALANCED),
}

DATA_FILE_NAMES_IMBALANCED = {
    TRAIN: '{}_data_{}.csv'.format(TRAIN, IMBALANCED),
    TEST: '{}_data_{}.csv'.format(TEST, IMBALANCED),
}

# Numeric Transfromation
SQRT = 'sqrt'
LOG_TRANSFORMATION= 'log'
QUAD = 'quad'


# Cadastral Quality Order
CADASTRAL_QUALITY_ORDER = ['9', '8', '7', '6', '5', '4', '3', '2', '1', 'C', 'B', 'A']

# Columns
NOT_TRANSFORMED_COLUMNS = [
    'X',
    'Y',
    'Q_R_4_0_0',
    'Q_R_4_0_1',
    'Q_R_4_0_2',
    'Q_R_4_0_3',
    'Q_R_4_0_4',
    'Q_R_4_0_5',
    'Q_R_4_0_6',
    'Q_R_4_0_7',
    'Q_R_4_0_8',
    'Q_R_4_0_9',
    'Q_R_4_1_0',
    'Q_G_3_0_0',
    'Q_G_3_0_1',
    'Q_G_3_0_2',
    'Q_G_3_0_3',
    'Q_G_3_0_4',
    'Q_G_3_0_5',
    'Q_G_3_0_6',
    'Q_G_3_0_7',
    'Q_G_3_0_8',
    'Q_G_3_0_9',
    'Q_G_3_1_0',
    'Q_B_2_0_0',
    'Q_B_2_0_1',
    'Q_B_2_0_2',
    'Q_B_2_0_3',
    'Q_B_2_0_4',
    'Q_B_2_0_5',
    'Q_B_2_0_6',
    'Q_B_2_0_7',
    'Q_B_2_0_8',
    'Q_B_2_0_9',
    'Q_B_2_1_0',
    'Q_NIR_8_0_0',
    'Q_NIR_8_0_1',
    'Q_NIR_8_0_2',
    'Q_NIR_8_0_3',
    'Q_NIR_8_0_4',
    'Q_NIR_8_0_5',
    'Q_NIR_8_0_6',
    'Q_NIR_8_0_7',
    'Q_NIR_8_0_8',
    'Q_NIR_8_0_9',
    'Q_NIR_8_1_0',
    'GEOM_R1',
    'GEOM_R2',
    'GEOM_R3',
    'GEOM_R4',
    'CONTRUCTIONYEAR',
]

# SENTINEL INDEXES
B8 = 'Q_NIR_8_0_5'
B4 = 'Q_R_4_0_5'
B3 = 'Q_G_3_0_5'
B2 = 'Q_B_2_0_5'

SAVI = 'SAVI'
PSSR = 'PSSR'
EVI = 'EVI'
EVI2 = 'EVI2'


SAVI_L = 0.483




TARGET_FEATURE = 'CLASE'
COORDINATES = ['X', 'Y']
GEOMS = ['GEOM_R1', 'GEOM_R2', 'GEOM_R3', 'GEOM_R4']
CADASTRAL_QUALITY = 'CADASTRALQUALITYID'
AREA = 'AREA'
BUILDING_YEAR = 'CONTRUCTIONYEAR'

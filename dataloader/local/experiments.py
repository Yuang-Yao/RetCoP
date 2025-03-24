from utils.constants import *


def get_experiment_setting(experiment):

    # Transferability for classification
    if experiment == "02_MESSIDOR":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "02_MESSIDOR.csv",
                   "task": "classification",
                   "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
                               "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
                               "proliferative diabetic retinopathy": 4}}
    elif experiment == "AMD":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "AMD.csv",
                   "task": "classification",
                   "targets": {"age related macular degeneration": 0, "normal": 1}}
    elif experiment == "TAOP":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "TAOP_train.csv",
                   "task": "classification",
                   "targets": {"0c":0, "1c":1, "2c":2, "3c":3, "4c":4}}
    elif experiment == "25_REFUGE":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "25_REFUGE.csv",
                   "task": "classification",
                   "targets": {"no glaucoma": 0, "glaucoma": 1}}
    elif experiment == "13_FIVES":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "13_FIVES.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic retinopathy": 2,
                               "glaucoma": 3}}
    elif experiment == "08_ODIR200x3":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "08_ODIR200x3.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}
    elif experiment == "05_20x3":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "05_20x3.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "retinitis pigmentosa": 1, "macular hole": 2}}
    elif experiment == "Angiographic":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION_FFA + "Angiographic.csv",
                   "task": "classification",
                   "targets": {"macular neovascularization": [0],
                               "unremarkable changes": [1], "dry age-related macular degeneration": [2],
                               "central serous chorioretinopathy": [3], "uveitis": [4], "chorioretinal scar": [5],
                               "diabetic retinopathy": [6], "retinal pigment epithelial detachment": [7],
                               "pachychoroid pigment epitheliopathy": [8], "chorioretinal atrophy": [9], "myopia": [10],
                               "proliferative diabetic retinopathy": [11], "cystoid macular edema": [12], "choroidal mass":[13],
                               "other": [14], "epiretinal membrane": [15], "retinal vein occlusion": [16],
                               "retinal arterial macroaneurysm":[17], "branch retinal vein occlusion": [18],
                               "central retinal vein occlusion": [19], "retinal dystrophy": [20], "polypoidal choroidal vasculopathy": [21],
                               "central retinal artery occlusion": [22] ,"cystoid macular edema,uveitis": [4, 12], "uveitis,retinal vein occlusion": [4, 16],
                               "choroidal mass,macular neovascularization": [0, 13], "diabetic retinopathy,macular neovascularization": [0, 6],
                               "macular neovascularization,myopia": [0, 10], "diabetic retinopathy,central serous chorioretinopathy": [3, 6],
                               "central serous chorioretinopathy,proliferative diabetic retinopathy": [3, 11],
                               "chorioretinal scar,diabetic retinopathy": [5, 6], "chorioretinal scar,myopia": [5, 10],
                               "macular neovascularization,branch retinal vein occlusion": [0, 18],
                               "macular neovascularization,polypoidal choroidal vasculopathy": [0, 21], "myopia,chorioretinal atrophy": [9, 10],
                               "myopia,dry age-related macular degeneration": [2, 10], "uveitis,central retinal artery occlusion": [4, 22]}}
    elif experiment == 'MPOS':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION_FFA + "MPOS.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "diabetic retinopathy": 1, "retinal vein occlusion": 2,
                               "age-related macular degeneration": 3, "central serous retinopathy": 4}}
    elif experiment == 'OCT05_OCTID':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION_OCT + "OCT05_OCTID.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "central serous retinopathy": 1, "age related macular degeneration": 2,
                               "macular hole": 3, "diabetic retinopathy": 4}}
    elif experiment == 'OCT10_OCTDL':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION_OCT + "OCT10_OCTDL.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic macular edema": 2,
                               "retinal artery occlusion": 3, "Vitreomacular Interface Disease": 4, "epiretinal membrane": 5, "retinal vein occlusion": 6}}

    else:
        setting = None
        print("Experiment not prepared...")

    return setting

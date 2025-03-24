"""
downstream task transferability for CFP dataset
"""

import argparse
import torch
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from models.model_cl import RetCoPModel
from downstream.data.dataloader import get_dataloader_splits
from utils.metrics import evaluate, average_folds_results, save_results
from utils.misc import set_seeds
from modeling.adapters import LinearProbe, ClipAdapter, ZeroShot, TipAdapter

from utils.constants import *
from dataloader.local.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")

set_seeds(42, use_cuda=torch.cuda.is_available())


#   initialize downstream task adapter
def init_adapter(model, args):
    # linear probe
    if args.method == "lp":
        print("Transferability by Linear Probing...", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)
    # CLIP  clip adapter
    elif args.method == "clipAdapter":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = ClipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge)
    # TipAdapter
    elif args.method == "tipAdapter":
        print("Transferability by TIP-Adapter Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge, train=False)
    # TipAdapter-f
    elif args.method == "tipAdapter-f":
        print("Transferability by TIP-Adapter-f Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge, train=True)
    # ZS  zero shot
    elif args.method == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge)
    # lp  By default Linear probe
    else:
        print("Adapter not implemented... using LP", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)

    return adapter


#   Get experiment name
def generate_experiment_id(args):
    id = args.experiment + '_name_' + args.architecture + '_method_' + args.method +\
         '_shots_train_' + args.shots_train + '_shots_test_' + args.shots_test + \
         '_balance_' + str(args.balance) + '_domain knowledge_' + str(args.domain_knowledge) + \
         '_proj_' + str(args.project_features)
    return id


def process(args):
    # metrics_test：  Test set evaluation metric list
    # metrics_external：，n  Additional data set evaluation metric list
    # weights：  Adapter weight list
    args.metrics_test, args.metrics_external, args.weights = [], [[] for i in range(len(args.experiment_test))], []     # experiment_test：  List of data set names

    experiment_id = generate_experiment_id(args)
    print(experiment_id)

    # K  K-fold cross-validation
    for iFold in range(args.folds):
        print("\nTransferability (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        #   data
        args.setting = get_experiment_setting(args.experiment)                                                          #  、、  Gets the experiment configuration: path, task, category
        args.loaders = get_dataloader_splits(args.setting["dataframe"], args.data_root_path, args.setting["targets"],
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, balance=args.balance,
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             task=args.setting["task"], size=args.size,
                                              batch_size_test=args.batch_size_test,
                                             knowledge_dict= args.knowledge_dict, dataset=args.experiment)                                       # （）；、、；dataloader
                                                                                                                        # Data transformation (preprocessing); Split training, valid and test sets; Create a dataloader

        #   model
        model = RetCoPModel(from_checkpoint=args.load_weights, weights_path=args.weights_path,
                             projection=args.project_features, norm_features=args.norm_features,
                             vision_pretrained=args.init_imagenet)
        adapter = init_adapter(model, args)                                                                             #   Initialize adapter

        #   training
        adapter.fit(args.loaders)

        #   prediction
        if args.loaders["test"] is not None:
            refs, preds = adapter.predict(args.loaders["test"])
            metrics_fold = evaluate(refs, preds, args.setting["task"])
            args.metrics_test.append(metrics_fold)

        # ，  Output, save adapter weight
        args.weights.append(adapter.model.state_dict())

        # OOD【 】  OOD Experiment Training on one data set and test for another Dataset
        # experiment_test：   input dataset    ZS
        if args.experiment_test[0] != "":
            #   Traverse the dataset
            for i_external in range(len(args.experiment_test)):
                print("External testing: " + args.experiment_test[i_external])

                #   data
                setting_external = get_experiment_setting(args.experiment_test[i_external])
                loaders_external = get_dataloader_splits(setting_external["dataframe"], args.data_root_path,
                                                         args.setting["targets"], shots_train="0%", shots_val="0%",
                                                         shots_test="100%", balance=False,
                                                         batch_size=args.batch_size_test,
                                                         batch_size_test=args.batch_size_test,
                                                         num_workers=args.num_workers, seed=iFold,
                                                         task=args.setting["task"], size=args.size,
                                                         resize_canvas=args.resize_canvas)
                #    Test data evaluation
                refs, preds = adapter.predict(loaders_external["test"])
                metrics = evaluate(refs, preds, args.setting["task"])
                args.metrics_external[i_external].append(metrics)

    # K  average K-fold
    if args.loaders["test"] is not None:
        print("\nTransferability (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test, args.setting["task"])
    else:
        args.metrics = None

    out_path = PATH_RESULTS_DOWNSTREAM + args.out_path + "/"
    # ，
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #    Save the evaluation metric and adapter weight
    save_results(args.metrics, out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)

    # OOD K    ood experiment  average K-fold
    if args.experiment_test[0] != "":
        for i_external in range(len(args.experiment_test)):
            print("External testing: " + args.experiment_test[i_external])
            metrics = average_folds_results(args.metrics_external[i_external], args.setting["task"])
            save_results(metrics, out_path, id_experiment=generate_experiment_id(args),
                         id_metrics=args.experiment_test[i_external], save_model=False)


def main():
    parser = argparse.ArgumentParser()

    #  data
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_DOWNSTREAM, help='output path')
    parser.add_argument('--experiment_description', default=None)
    parser.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))       #   save the adaptation weight?
    parser.add_argument('--shots_train', default="0%", type=lambda x: (str(x)))                       #   proportion of data used for training
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))                           #   data used for validation
    parser.add_argument('--shots_test', default="20%", type=lambda x: (str(x)))                          # ZS  data used for testing
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))          #   Balanced dataset?
    parser.add_argument('--folds', default=1, type=int)                                                 # K  K-fold cross validation
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    parser.add_argument('--size', default=(512, 512), help="(512, 512) | (2048, 4096) ")
    parser.add_argument('--resize_canvas', default=False, type=lambda x: (str(x).lower() == 'true'))    

    #   Experimental parameter
    parser.add_argument('--experiment', default='08_ODIR200x3',
                        help='02_MESSIDOR - 13_FIVES - 25_REFUGE - 08_ODIR200x3 - 05_20x3 - AMD - TAOP -'
                             'Angiographic - MPOS') #   data set used in the experiment
    parser.add_argument('--experiment_test', default='',
                        help='02_MESSIDOR, 37_DeepDRiD_online_test',
                        type=lambda s: [item for item in s.split(',')])                                             # OOD
    parser.add_argument('--method', default='zero_shot',
                        help='lp - tipAdapter - tipAdapter-f - clipAdapter'
                             'FT - FT_last - LP_FT -LP_FT_bn_last - FT_freeze_all'
                             'zero_shot -')                                                             #  ZS、FT、LP  adaptation method
    parser.add_argument('--num_workers', default=24, type=int, help='workers number for DataLoader')
    parser.add_argument('--test_from_folder', default=[], type=list)

    parser.add_argument('--epochs', default=50, type=int)                                               # FT  FT setting
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--update_bn', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--freeze_classifier', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--last_lp', default=False, type=lambda x: (str(x).lower() == 'true'))          # FTLP  FT with lp
    parser.add_argument('--save_best', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--patience', default=10, type=int)

    #   Model architecture and weight
    parser.add_argument('--weights_path', default='/mnt/data_ssd/yayao/CPT_retinal/RetCoP/checkpoint/CFP_epoch20.pth',
                        help='/mnt/data_ssd/yayao/CPT_retinal/RetCoP/checkpoint')                     #    local weights, downloaded by default
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))      #  True   load pre-trained weight?
    parser.add_argument('--init_imagenet', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--architecture', default='KD', help='resnet_v1 -- efficientnet')
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true'))  # 
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))     # 
    parser.add_argument('--domain_knowledge', default=True, type=lambda x: (str(x).lower() == 'true'))  #   use domain knowledge when adaptation?
    parser.add_argument('--fta', default=False, type=lambda x: (str(x).lower() == 'true'))              #   train time data augmentation
    parser.add_argument('--tta', default=False, type=lambda x: (str(x).lower() == 'true'))              #   test time data augmentation
    parser.add_argument('--knowledge_dict', default=False, type=lambda x: (str(x).lower() == 'true'))   # caption  use domain knowledge caption?

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()

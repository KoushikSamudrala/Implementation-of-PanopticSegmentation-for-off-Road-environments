# importing the required modules
import os
import subprocess
import json
import random
import cv2
import json
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.engine.defaults import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config, PanopticDeeplabDatasetMapper # noqa
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.evaluation import COCOPanopticEvaluator, COCOEvaluator, DatasetEvaluators

Vis = True
Train = False
Infer = False


categories_file_path = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/train/Categories.json"
with open(categories_file_path, "r") as json_file:
    categories_data = json.load(json_file)

stuff_names = [cat["name"] for cat in categories_data]
thing_names = [cat["name"] for cat in categories_data]

stuff_ids = [cat["id"] for cat in categories_data if cat["isthing"]==0]
thing_ids = [cat["id"] for cat in categories_data if cat["isthing"]==1]

total_classes = len(stuff_ids) + len(thing_ids)

# stuff_contigous_ids = { id:i for i, id in enumerate(stuff_ids)}
# thing_contigous_ids = { id:i for i, id in enumerate(thing_ids)}


stuff_contigous_ids = dict((zip(stuff_ids, list(range(0, len(stuff_ids))))))
thing_contigous_ids = dict(zip(thing_ids,list(range(len(stuff_ids), total_classes))))

# thing_contigous_ids = dict(zip(thing_ids,list(range(0, len(thing_ids)))))
# stuff_contigous_ids = dict(zip(stuff_ids, list(range(len(thing_ids), total_classes))))

print("+"*50)

print("stuff_classes:", stuff_names)
print("stuff_ids:", stuff_ids)
print("thing_classes:", thing_names)
print("thing_ids", thing_ids)


metadata_dict = { "stuff_classes": stuff_names,
                 "thing_classes": thing_names,
                 "thing_dataset_id_to_contiguous_id": thing_contigous_ids,
                 "stuff_dataset_id_to_contiguous_id": stuff_contigous_ids}

print("+"*50)
print(metadata_dict)


data_name = 'ZED_cam_data_sample_training_panoptic'

data_root_dir = '/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/'

for i in ["train", "val"]:

    root_dir = os.path.join(data_root_dir, i + '/' + i + "_gt")
    assert os.path.isdir(root_dir), "Directory does not exist"
    image_root = os.path.join(root_dir, "JPEGImages")
    panoptic_root = os.path.join(root_dir, "coco_panoptic")
    panoptic_json = os.path.join(root_dir, "coco_panoptic.json")
    with open(panoptic_json) as panoptic_json_file:
        panoptic_dict = json.load(panoptic_json_file)
    sem_seg_root = os.path.join(root_dir, "panoptic_semantic_masks")
    instances_json = os.path.join(root_dir, "coco_panoptic_instance.json")
    register_coco_panoptic(data_name + i, metadata_dict, image_root, panoptic_root, panoptic_json,
                           instances_json)
    print("+"*50)
    dataset_dicts = DatasetCatalog.get(data_name + i)
    print("sample loaded dataset dict:", dataset_dicts[0])

train_metadata = MetadataCatalog.get(data_name + "train")


if Vis:

    for i in ["train", "val"]:
        dataset_dicts = DatasetCatalog.get(data_name + i )
        for d in random.sample(dataset_dicts, 5):
            print(f'visualizing {i} dataset' )
            img = cv2.imread(d["file_name"])
            print(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata= train_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow("Image", out.get_image()[:, :, ::-1])

            # Wait for a key press
            cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()


config_file = "/home/yalamaku/Documents/Detectron_2/detectron2/projects/Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml"

cfg = get_cfg()

add_panoptic_deeplab_config(cfg)

cfg.merge_from_file(config_file)

cfg.DATASETS.TRAIN = (data_name + "train",)
cfg.DATASETS.TEST = (data_name + "val",)
cfg.TEST.EVAL_PERIOD = 50
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.DEVICE='cpu'

cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Detectron_2/downloaded_weights/COCO_dsconv.pkl"
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 2000

cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = total_classes-1
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.OUTPUT_DIR = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/Sample_train_weights/Panoptic_DeepLab_coco_4"

if Train:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok= True)

    def build_sem_seg_train_aug(cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())
        return augs

    class CustomTrainer(DefaultTrainer):

        @classmethod
        def build_train_loader(cls, cfg):
            mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
            return build_detection_train_loader(cfg, mapper=mapper)
        
        """Overwriting Build_lr_scheduler class method of base class DefaultTrainer"""
        @classmethod
        def build_lr_scheduler(cls, cfg, optimizer):
            """
            It now calls :func:`detectron2.solver.build_lr_scheduler`.
            Overwrite it if you'd like a different scheduler.
            """
            return build_lr_scheduler(cfg, optimizer)
        

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):

            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

            evaluator_list = []
            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

            if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

            if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.

                evaluator_list.append(
                    COCOEvaluator(dataset_name, output_dir=output_folder)
                )

            if len(evaluator_list) == 0:
                raise NotImplementedError(
                    "no Evaluator for the dataset {} with the type {}".format(
                        dataset_name, evaluator_type
                    )
                )
            elif len(evaluator_list) == 1:
                return evaluator_list[0]
            return DatasetEvaluators(evaluator_list)
        
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume= False)
    trainer.train()



if Infer:
    # Check result
    cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/Sample_train_weights/Panoptic_DeepLab_coco_4/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    predictor = DefaultPredictor(cfg)
    # for single test image
    im = cv2.imread("/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/ZED_cam_images/1221.png")
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], train_metadata, scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

    # # panoptic segmentation result
    # plt.imshow(v.get_image())
    # plt.show()

    # Save the image to a folder
    output_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/infer_out_put/Panoptic_DeepLab/Panoptic_DeepLab_4"  # Specify the desired output folder
    output_path = os.path.join(output_folder, "panoptic_seg_result_7.png")
    cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
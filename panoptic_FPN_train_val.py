import os, subprocess
import json
import random
import cv2
import json
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, COCOPanopticEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader

Vis = False
Train = False
Eval = False
Infer = True

categories_file_path = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/train/Categories.json"
with open(categories_file_path, "r") as json_file:
    categories_data = json.load(json_file)


# Reason for considering stuff_names = both stuff_names and thing_names (keyerror: 7 and 8 are out of bounds).
# you might get key error of 7 and 8: beacuse while performing evaluation the detectron2 evaluation scripts use panoptic_groud_truth masks (coco_panoptic)
# which contains both stuff and things (7,8). which will lead to an error
# 7 and 8 are ids of things which are presnt in both ground truth of panoptic and semantic masks of ground truth
# the panoptic ground truth contains both stuff and things 
# if you need only stuff names to be in stuff. then while generating panoptic_semantic_masks use label --things_other
# which will convert all things into a single semantic mask
stuff_names = [cat["name"] for cat in categories_data]
thing_names = [cat["name"] for cat in categories_data if cat["isthing"]==1]

stuff_ids = [cat["id"] for cat in categories_data]
thing_ids = [cat["id"] for cat in categories_data if cat["isthing"]==1]

stuff_contigous_ids = { id:i for i, id in enumerate(stuff_ids)}
thing_contigous_ids = { id:i for i, id in enumerate(thing_ids)}

total_classes = len(stuff_ids) + len(thing_ids)

# thing_contigous_ids = dict(zip(thing_ids,list(range(len(stuff_ids), total_classes))))
# stuff_contigous_ids = dict((zip(stuff_ids, list(range(0, len(stuff_ids))))))

# thing_contigous_ids = dict(zip(thing_ids,list(range(0, len(thing_ids)))))
# stuff_contigous_ids = dict(zip(stuff_ids, list(range(len(thing_ids), total_classes))))


print("stuff_classes:", stuff_names)
print("stuff_ids:", stuff_ids)
print("thing_classes:", thing_names)
print("thing_ids", thing_ids)

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
    register_coco_panoptic_separated(data_name + i, {}, image_root, panoptic_root, panoptic_json, sem_seg_root,
                                    instances_json)
    dataset_dicts = DatasetCatalog.get(data_name + i + "_separated")
    print("sample loaded dataset dict:", dataset_dicts[0])
    metadata = MetadataCatalog.get(data_name + i +"_separated").set(stuff_classes = stuff_names, 
                                                                   thing_classes = thing_names, 
                                                                   thing_dataset_id_to_contiguous_id = thing_contigous_ids,
                                                                   stuff_dataset_id_to_contiguous_id = stuff_contigous_ids)

train_metadata = MetadataCatalog.get(data_name + "train_separated")


if Vis:

    for i in ["train", "val"]:
        dataset_dicts = DatasetCatalog.get(data_name + i + "_separated")
        for d in random.sample(dataset_dicts, 5):
            print(f'visualizing {i} dataset' )
            img = cv2.imread(d["file_name"])
            print(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata= metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow("Image", out.get_image()[:, :, ::-1])

            # Wait for a key press
            cv2.waitKey(0)


        # Close all windows
        cv2.destroyAllWindows()



config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.DATASETS.TRAIN = (data_name + "train_separated",)
cfg.DATASETS.TEST = (data_name + "val_separated",)
cfg.TEST.EVAL_PERIOD = 10
cfg.DATALOADER.NUM_WORKERS = 2   
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo

# solver 
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 350  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset


cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_metadata.thing_classes)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(train_metadata.stuff_classes)
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.OUTPUT_DIR = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/Sample_train_weights/Panoptic_FPN_R50_3X_2"



if Train:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    class CustomTrainer(DefaultTrainer):
        
        @classmethod
        def build_evaluator(cls, cfg, dataset_name):

            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

            evaluator_list = []
            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

            if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

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



#Check result
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

if Eval:
    
    val_loader = build_detection_test_loader(cfg, data_name+"val_separated")

    print("COCO_Eavaluator results")
    evaluator = COCOEvaluator(dataset_name= data_name + "val_separated",
                              output_dir= cfg.OUTPUT_DIR)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    print("COCO_Panoptic_Evaluator")
    panoptic_evaluator = COCOPanopticEvaluator(dataset_name= data_name + "train_separated", output_dir= cfg.OUTPUT_DIR)
    print(inference_on_dataset(predictor.model, val_loader, panoptic_evaluator))


if Infer:
    # for single test image
    im = cv2.imread("/home/yalamaku/Documents/Detectron_2/detectron2/datasets/coco/val2017/000000158548.jpg")
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], train_metadata, scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

    # panoptic segmentation result
    # plt.imshow(v.get_image())
    # plt.show()

    # Save the image to a folder
    output_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_cam_data/my_dataset/sample_dataset/infer_out_put/Panoptic_FPN/ Panoptic_FPN_R50_3x_2"  # Specify the desired output folder
    output_path = os.path.join(output_folder, "panoptic_seg_result_9.png")
    cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
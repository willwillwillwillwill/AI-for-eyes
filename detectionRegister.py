from detectron2.data.datasets import register_coco_instances
register_coco_instances("eye_train",
                        {"thing_classes": ["Paint crack", "CNV", "Fuchs"],
                         "thing_colors": [[0, 255, 0], [0, 255, 0], [0, 255, 0]]}
                        ,
                        "./data/eye_images/instances_train.json",
                        "./data/eye_images/train")
register_coco_instances("eye_test",
                        {"thing_classes":["Paint crack", "CNV", "Fuchs"],"thing_colors":[[0,255,0],[0,255,0],[0,255,0]]},
                        "./data/eye_images/instances_val.json",
                        "./data/eye_images/val")
# register_coco_instances("eye_test",
#                         {"thing_classes":["Paint crack", "CNV", "Fuchs"],"thing_colors":[[0,255,0],[0,255,0],[0,255,0]]},
#                         "./data/eye_images/new_instances_val.json",
#                         "/data1/lulixian/test")
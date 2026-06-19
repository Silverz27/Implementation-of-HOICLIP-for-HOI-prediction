import kagglehub

# Download latest version
path = kagglehub.dataset_download("sliverz/meva-datasets")

print("Path to dataset files:", path)

#Train
#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --resume checkpoint_last.pth --dataset_file hico --hoi_path D:/meva_datasets/datasets --num_obj_classes 3 --num_verb_classes 8 --backbone resnet50 --num_queries 64 --dec_layers 3 --epochs 30 --lr_drop 20 --batch_size 8 --num_workers 8 --with_clip_label --with_obj_clip_label --use_nms_filter --output_dir D:/meva_datasets/train_results

#Eval
#python main.py --resume checkpoint_last.pth --dataset_file hico --hoi_path (dataset path) --num_obj_classes 3 --num_verb_classes 8 --backbone resnet50 --num_queries 64 --dec_layers 3 --eval --num_workers 8 --with_clip_label --with_obj_clip_label --use_nms_filter --output_dir (output dir)


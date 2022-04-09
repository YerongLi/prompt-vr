This repo is upgraded version of [1] without detection module:

[1] https://github.com/NVIDIA/ContrastiveLosses4VRD

## Requirements
* Python 3.9
* Python packages
* An NVIDIA GPU and CUDA 9.0 or higher. Some operations only have gpu implementation.

An easy installation if you already have Anaconda Python 3.9 and CUDA>=9.0:
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio cudatoolkit=10.2 -c pytorch
pip install cython
pip install matplotlib numpy scipy pyyaml packaging tensorboardX tqdm pillow scikit-image
conda install opencv
```

## Compilation
GCC 5.4 (mutillib enabled) GLIBC 2.31 CUDA 11.1
```
# ROOT=path/to/cloned/repository
cd $ROOT/lib
sh make.sh
```


## Images and Anotations

### Visual Relation Detection
Create the vrd folder under `data`:
```
cd $HOME/data/vrd
```
Download the original annotation json files from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and unzip `json_dataset.zip` here. The images can be downloaded from [here](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip). Unzip `sg_dataset.zip` to create an `sg_dataset` folder in `data/vrd`. Next run the preprocessing scripts:

```
cd $ROOT
python tools/rename_vrd_with_numbers.py
python tools/convert_vrd_anno_to_coco_format.py
```
`rename_vrd_with_numbers.py` converts all non-jpg images (some images are in png or gif) to jpg, and renames them in the {:012d}.jpg format (e.g., "000000000001.jpg"). It also creates new relationship annotations other than the original ones. This is mostly to make things easier for the dataloader. The filename mapping from the original is stored in `data/vrd/*_fname_mapping.json` where "*" is either "train" or "val".

`convert_vrd_anno_to_coco_format.py` creates object detection annotations from the new annotations generated above, which are required by the dataloader during training.

### Visual Genome (TODO)

## Directory Structure
The final directories for data should look like:
```
|-- data
|   |-- vrd
|   |   |-- train_images    <-- (contains Visual Relation Detection training images)
|   |   |-- val_images    <-- (contains Visual Relation Detection validation images)
|   |   |-- new_annotations_train.json
|   |   |-- new_annotations_val.json
|   |   |-- ...
```


### Visual Genome (TODO)

## Evaluating Pre-trained Relationship Detection models

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to test with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use. Remove the
`--multi-gpu-test` Is disabled.



### Visual Relation Detection
To test a evaluation a trained model run
```
git pull;cd lib;sh make.sh;cd ..;CUDA_VISIBLE_DEVICES=6 python ./tools/test_net_rel.py --dataset vrd --cfg configs/vrd/e2e_faster_rcnn_VGG16_16_epochs_vrd_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_IN_pretrained.yaml --load_ckpt ~/trained_models/vrd_VGG16_IN_pretrained/model_step7559.pth --output_dir Outputs/vrd_VGG16_IN_pretrained --do_val --use_gt_boxesgit pull;cd lib;sh make.sh;cd ..;CUDA_VISIBLE_DEVICES=6 python ./tools/test_net_rel.py --dataset vrd --cfg configs/vrd/e2e_faster_rcnn_VGG16_16_epochs_vrd_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_IN_pretrained.yaml --load_ckpt ~/trained_models/vrd_VGG16_IN_pretrained/model_step7559.pth --output_dir Outputs/vrd_VGG16_IN_pretrained --do_val --use_gt_boxes
```
See [img/load_vrd.jpg](img/load_vrd.jpg) for expected output, as long as we can see bounding box we are good.


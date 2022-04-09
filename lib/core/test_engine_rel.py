# Adapted by Ji Zhang, 2019
# from Detectron.pytorch/lib/core/test_engine.py
# Original license text below
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
from numpy import linalg as la
import os
import yaml
import json
from six.moves import cPickle as pickle


from core.config import cfg
from core.test_rel import im_detect_rels
from datasets_rel import task_evaluation_vg_and_vrd as task_evaluation_vg_and_vrd
from datasets_rel.json_dataset_rel import JsonDatasetRel
import utils_rel.subprocess_rel as subprocess_utils
from utils.io import save_object
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    # Generic case that handles all network types other than RPN-only nets
    # and RetinaNet
    child_func = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]
    proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None
    logger.info('Entering run_inference')
    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            logger.info('Is Parent case')
            all_results = []
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                # print(dataset_name)
                # vrd_val
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.append(results)
                # print('results')
                # print(results)
            return all_results
        else:
            # Have to disable child branch temporarily
            raise NotImplementedError
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDatasetRel(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb(gt=args.do_val))
        all_results = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        logger.info('Entering the branch of test_net')
        all_results = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    
    logger.info('Starting evaluation now...')
    if dataset_name.find('vg') >= 0 or dataset_name.find('vrd') >= 0:
        task_evaluation_vg_and_vrd.eval_rel_results(all_results, output_dir, args.do_val)
    # else:
    #     task_evaluation_sg.eval_rel_results(all_results, output_dir, args.do_val, args.do_vis, args.do_special)
    
    return all_results

def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    raise NotImplemented
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]
        
    if args.do_val:
        opts += ['--do_val']
    if args.do_vis:
        opts += ['--do_vis']
    if args.do_special:
        opts += ['--do_special']
    if args.use_gt_boxes:
        opts += ['--use_gt_boxes']
    if args.use_gt_labels:
        opts += ['--use_gt_labels']

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'rel_detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_results = []
    for det_data in outputs:
        all_results += det_data
    
    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')
    save_object(all_results, det_file)
    logger.info('Wrote rel_detections to: {}'.format(os.path.abspath(det_file)))

    return all_results


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    logger.info('Entering test_net')
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range, args.do_val
    )
    # print('Loaded bounding box in the dataset', list(dataset.rel_anns.items())[:2])
    # What is the model
    # model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    model = None
    num_images = len(roidb)
    all_results = [None for _ in range(num_images)]
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        box_proposals = None
        ## Read the image
        im = cv2.imread(entry['image'])
        logger.info(entry['image'])
        if True or args.use_gt_boxes:## DEBUG
            im_results = im_detect_rels(model, im, dataset_name, box_proposals, args.do_vis, timers, entry, args.use_gt_labels)
        else:
            im_results = im_detect_rels(model, im, dataset_name, box_proposals, args.do_vis, timers)
        logger.info('im_results')

        # print(im_results.keys())
        # dict_keys(['sbj_boxes', 'sbj_labels', 'sbj_scores', 'obj_boxes', 'obj_labels', 'obj_scores', 'prd_scores', 'prd_scores_ttl', 'prd_scores_bias', 'prd_s
# cores_spt'])
        print(im_results['sbj_labels'])
        print(im_results['sbj_boxes'])
        # [3 3 8 8 0 0 0 0 8 8 8]


        # --use_gt_boxes
        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000002.jpg                                              
        # INFO test_engine_rel.py: 246: im_results 
        #                                                                                                              
        # [65  5  0]                                                                                                                                            
        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000003.jpg                                              
        # INFO test_engine_rel.py: 246: im_results
        # [ 0  0  0  0 10 95]
        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000004.jpg
        # INFO test_engine_rel.py: 246: im_results
        # [ 4  8 21]

        # --use_gt_boxes

        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000001.jpg                                                                                                                [18/1795]
        # /scratch/yerong/local/anaconda3/envs/vr1/lib/python3.6/site-packages/torch/nn/functional.py:1006: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.                                                   
        # warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")                                                                    
        # INFO test_engine_rel.py: 246: im_results                                                                                                              
        # [2]                                                                                                                                                   
        # [[  8. 456. 101. 487.]]                                                                                                                               
        # INFO test_engine_rel.py: 289: im_detect: range [1, 1000] of 1000: 1/1000 0.438s (eta: 0:07:17)                                                        
        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000002.jpg                                              
        # INFO test_engine_rel.py: 246: im_results                                                                                                              
        # [65  5  0]                                                                                                                                            
        # [[ 583.  410.  714.  525.]                                                                                                                            
        # [ 482.  265.  891.  762.]                                                                                                                            
        # [ 660.  440. 1020.  767.]]                                                                                                                           
        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000003.jpg                                                                                                                         
        # INFO test_engine_rel.py: 246: im_results                                                                                                              
        # [ 0  0  0  0 10 95]                                                                                                                                                      
        # [[1.000e+00 1.010e+02 1.100e+02 6.010e+02]                                          
        # [1.000e+00 1.010e+02 1.100e+02 6.010e+02]                                                      
        # [7.070e+02 2.410e+02 8.460e+02 5.500e+02]                                                      
        # [8.650e+02 2.670e+02 1.024e+03 6.750e+02]                                                      
        # [9.530e+02 3.200e+02 1.021e+03 3.610e+02]                                                                                                            
        # [9.600e+02 4.920e+02 9.960e+02 5.540e+02]]                                                                                                                              
        # INFO test_engine_rel.py: 241: /scratch/yerong/ContrastiveLosses4VRD/data/vrd/val_images/000000000004.jpg                                                                                                                         
        # INFO test_engine_rel.py: 246: im_results                                                                        
        # [ 4  8 21]                                      
        # [[329. 323. 650. 519.]                          
        # [719. 422. 774. 455.]                                                                                                                                                                           
        # [494. 455. 524. 520.]]     

        im_results.update(dict(image=entry['image']))
        # add gt
        if args.do_val:
            im_results.update(
                dict(gt_sbj_boxes=entry['sbj_gt_boxes'],
                     gt_sbj_labels=entry['sbj_gt_classes'],
                     gt_obj_boxes=entry['obj_gt_boxes'],
                     gt_obj_labels=entry['obj_gt_classes'],
                     gt_prd_labels=entry['prd_gt_classes']))
        
        all_results[i] = im_results

        if i % 10 == 0:  # Reduce log file size
            # logger.info('Always load the ground truth')
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (timers['im_detect_rels'].average_time)
            logger.info((
                'im_detect: range [{:d}, {:d}] of {:d}: '
                '{:d}/{:d} {:.3f}s (eta: {})').format(
                start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                start_ind + num_images, det_time, eta))

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'rel_detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        if args.use_gt_boxes:
            if args.use_gt_labels:
                det_name = 'rel_detections_gt_boxes_prdcls.pkl'
            else:
                det_name = 'rel_detections_gt_boxes_sgcls.pkl'
        else:
            det_name = 'rel_detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(all_results, det_file)
    logger.info('Wrote rel_detections to: {}'.format(os.path.abspath(det_file)))
    logger.info('resturn all_results')
    return all_results

def get_roidb_and_dataset(dataset_name, proposal_file, ind_range, do_val=True):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    Param dataset_name DATASETS.keys in file lib/datasets_rel/dataset_catalog_rel.py
    Return roidb
    Return loaced dataset from the original datafiles
            dataset.rel_anns contains the bounding boxes inside each image
    """
    logger.info('Load dataset with annotations with JsonDatasetRel')
    dataset = JsonDatasetRel(dataset_name)
    logger.info('Last time loading the dataset')
    logger.info(dataset.rel_anns['000000000002.jpg'])
    # [{'predicate': 0, 'object': {'category': 40, 'bbox': [265, 762, 482, 891]}, 
    # 'subject': {'category': 65, 'bbox': [410, 525, 583, 714]}}, 
    # {'predicate': 2, 'object': {'category': 65, 'bbox': [410, 525, 583, 714]}, 
    # 'subject': {'category': 40, 'bbox': [265, 762, 482, 891]}}, 
    # {'predicate': 27, 'object': {'category': 40, 'bbox': [265, 762, 482, 891]}, 
    # 'subject': {'category': 0, 'bbox': [440, 767, 660, 1020]}}]
    roidb = dataset.get_roidb(gt=do_val)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images

import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from .utils import (
    remap_pose,
    get_single_bop_annotation,
    load_bop_meshes,
    load_bbox_3d,
    compute_pose_diff,
    compute_pose_diff_speed
)

from .poses import PoseAnnot

def evaluate(cfg, predictions):
    INF = 100000000
    classNum = cfg['data']['N_CLASS'] - 1 # get rid of the background class

    thresholds_adi = [0.05, 0.10, 0.20, 0.50]
    thresholds_rep = [2, 5, 10, 20]
        
    accuracy_adi_per_class = []
    accuracy_rep_per_class = []
    # 
    depth_bins = 3
    accuracy_adi_per_depth = []
    accuracy_rep_per_depth = []

    meshes, _ = load_bop_meshes(cfg['data']['MESH_DIR'])
    meshDiameter = cfg['data']['MESH_DIAMETERS']
    surfacePts = []
    for ms in meshes:
        pts = np.array(ms.vertices)
        tmp_index = np.random.choice(len(pts), 1000, replace=True)
        pts = pts[tmp_index]
        surfacePts.append(pts)

    keypoints_3d = load_bbox_3d(cfg['data']['BBOX_FILE'])
    predictions_for_eval = remap_predictions(
        np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3),
        cfg['INPUT']['INTERNAL_WIDTH'],
        cfg['INPUT']['INTERNAL_HEIGHT'],
        keypoints_3d,
        predictions
    )

    # get depth range from annotations, and divide it to serval bins
    depth_min = INF
    depth_max = 0
    for filename, item in predictions_for_eval.items():
        gt = item['gt']
        for clsid, R, T in gt:
            depth = float(T[2])
            depth_min = min(depth_min, depth)
            depth_max = max(depth_max, depth)
    depth_max += 1e-5 # add some margin for safe depth index computation
    depth_bin_width = (depth_max - depth_min) / depth_bins

    errors_adi_per_depth = list([] for i in range(0, depth_bins))
    errors_rep_per_depth = list([] for i in range(0, depth_bins))
    for clsid in range(classNum):
        errors_adi_all = [] # 3D errors
        errors_rep_all = [] # 2D errors
        depth_all = [] # depth for each sample
        # 
        for filename, item in predictions_for_eval.items():
            K = item['K']
            pred = item['pred']
            gt = item['gt']
            
            # filter by class id
            pred = [p for p in pred if p[1] == clsid]
            gt = [g for g in gt if g[0] == clsid]
            if len(gt) == 0:
                continue

            # find predictions with best confidences
            assert(len(gt) == 1) # only one object for one class now

            # get the depth bin of the object
            depth = float(gt[0][2][2])
            depth_idx = int((depth - depth_min) / depth_bin_width)
            depth_all.append(depth)
            # 
            if len(pred) > 0:
                # find the best confident one
                bestIdx = 0
                R1 = gt[0][1]
                T1 = gt[0][2]
                R2 = pred[bestIdx][2]
                T2 = pred[bestIdx][3]
                err_3d, err_2d = compute_pose_diff(surfacePts[clsid], K, R1, T1, R2, T2)
                # 
                errors_adi_all.append(err_3d / meshDiameter[clsid])
                errors_rep_all.append(err_2d)
                errors_adi_per_depth[depth_idx].append(err_3d / meshDiameter[clsid])
                errors_rep_per_depth[depth_idx].append(err_2d)
            else:
                errors_adi_all.append(1.0)
                errors_rep_all.append(50)
                errors_adi_per_depth[depth_idx].append(1.0)
                errors_rep_per_depth[depth_idx].append(50)

        assert(len(errors_adi_all) == len(errors_rep_all))
        counts_all = len(errors_adi_all)
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_all) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_class.append(accuracy)
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_all) < th).sum()
                accuracy[('REP%02dpx'%th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_class.append(accuracy)
        else:
            accuracy_adi_per_class.append({})
            accuracy_rep_per_class.append({})
    # 
    # compute accuracy for every depth bin
    for i in range(depth_bins):
        assert(len(errors_adi_per_depth[i]) == len(errors_rep_per_depth[i]))
        counts_all = len(errors_adi_per_depth[i])
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_per_depth[i]) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_depth.append(accuracy)
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_per_depth[i]) < th).sum()
                accuracy[('REP%02dpx'%th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_depth.append(accuracy)
        else:
            accuracy_adi_per_depth.append({})
            accuracy_rep_per_depth.append({})
    # 
    return accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, [depth_min, depth_max]


def evaluate_pose_predictions(predictions, class_number, meshes, mesh_diameters, symmetry_types):
    INF = 100000000
    classNum = class_number - 1  # get rid of the background class

    thresholds_adi = [0.05, 0.10, 0.20,0.50]
    thresholds_rep = [2, 5, 10, 20]

    accuracy_adi_per_class = []
    accuracy_rep_per_class = []
    #
    depth_bins = 3
    accuracy_adi_per_depth = []
    accuracy_rep_per_depth = []

    surfacePts = []
    for ms in meshes:
        pts = np.array(ms.vertices)
        tmp_index = np.random.choice(len(pts), 1000, replace=True)
        pts = pts[tmp_index]
        surfacePts.append(pts)

    # get depth range from annotations, and divide it to serval bins
    depth_min = INF
    depth_max = 0
    for filename, item in predictions.items():
        gtTs = np.array(item['meta']['translations'])
        for T in gtTs:
            depth = float(T.reshape(-1)[2])
            depth_min = min(depth_min, depth)
            depth_max = max(depth_max, depth)
    depth_max += 1e-5  # add some margin for safe depth index computation
    depth_bin_width = (depth_max - depth_min) / depth_bins

    errors_adi_per_depth = list([] for i in range(0, depth_bins))
    errors_rep_per_depth = list([] for i in range(0, depth_bins))
    for clsid in range(classNum):
        isSym = (("cls_" + str(clsid)) in symmetry_types)
        errors_adi_all = []  # 3D errors
        errors_rep_all = []  # 2D errors
        errors_speed_all = []  # in speed metric
        depth_all = []  # depth for each sample
        object_cx_all = []
        object_cy_all = []
        #
        for filename, item in predictions.items():
            K = np.array(item['meta']['K'])
            pred = item['pred']
            gtIDs = item['meta']['class_ids']
            gtRs = np.array(item['meta']['rotations'])
            gtTs = np.array(item['meta']['translations'])

            # filter by class id
            pred = [p for p in pred if p[1] == clsid]
            gtIdx = [gi for gi in range(len(gtIDs)) if gtIDs[gi] == clsid]
            if len(gtIdx) == 0:
                continue

            # find predictions with best confidences
            assert (len(gtIdx) == 1)  # only one object for one class now

            # get the depth bin of the object
            gi = gtIdx[0]  # only pick up the first one
            depth = float(gtTs[gi].reshape(-1)[2])
            depth_idx = int((depth - depth_min) / depth_bin_width)
            depth_all.append(depth)
            #
            if len(pred) > 0:
                # find the best confident one
                bestIdx = 0
                R1 = gtRs[gi]
                T1 = gtTs[gi]
                R2 = np.array(pred[bestIdx][2])
                T2 = np.array(pred[bestIdx][3])
                err_3d, err_2d = compute_pose_diff(surfacePts[clsid], K, R1, T1, R2, T2, isSym=isSym)
                #
                err_r, err_t = compute_pose_diff_speed(R1, T1, R2, T2)
                errors_speed_all.append(err_r + err_t)
                #
                # get the reprojected center
                tmp_pt = np.matmul(K, T1)
                object_cx = tmp_pt[0] / tmp_pt[2]
                object_cy = tmp_pt[1] / tmp_pt[2]
                object_cx_all.append(float(object_cx))
                object_cy_all.append(float(object_cy))
                #
                errors_adi_all.append(err_3d / mesh_diameters[clsid])
                errors_rep_all.append(err_2d)
                errors_adi_per_depth[depth_idx].append(err_3d / mesh_diameters[clsid])
                errors_rep_per_depth[depth_idx].append(err_2d)
            else:
                object_cx_all.append(-1)
                object_cy_all.append(-1)
                errors_adi_all.append(1.0)
                errors_rep_all.append(50)
                errors_adi_per_depth[depth_idx].append(1.0)
                errors_rep_per_depth[depth_idx].append(50)
        #
        err_vs_pos = np.stack((np.array(object_cx_all), np.array(object_cy_all), np.array(errors_adi_all))).transpose()
        dis_to_center = np.sqrt(
            (np.array(object_cx_all) - 512) * (np.array(object_cx_all) - 512) + (np.array(object_cy_all) - 512) * (
                        np.array(object_cy_all) - 512))
        erro_vs_center_dis = np.stack((dis_to_center, np.array(errors_adi_all))).transpose()
        # np.savetxt('out.txt', erro_vs_center_dis, fmt='%.3f')

        assert (len(errors_adi_all) == len(errors_rep_all))
        counts_all = len(errors_adi_all)
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_all) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_class.append(accuracy)
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_all) < th).sum()
                accuracy[('REP%02dpx' % th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_class.append(accuracy)
        else:
            accuracy_adi_per_class.append({})
            accuracy_rep_per_class.append({})
    #
    # compute accuracy for every depth bin
    for i in range(depth_bins):
        assert (len(errors_adi_per_depth[i]) == len(errors_rep_per_depth[i]))
        counts_all = len(errors_adi_per_depth[i])
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_per_depth[i]) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_depth.append(accuracy)
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_per_depth[i]) < th).sum()
                accuracy[('REP%02dpx' % th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_depth.append(accuracy)
        else:
            accuracy_adi_per_depth.append({})
            accuracy_rep_per_depth.append({})
    # print(sum(errors_speed_all)/len(errors_speed_all))
    return accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, [depth_min, depth_max]


def remap_predictions(internal_K, internal_width, internal_height, keypoints_3d, predictions):
    new_results = {}
    for imagename in predictions:
        # print(imagename)
        meta = predictions[imagename]['meta']
        pred = predictions[imagename]['pred']

        K = meta['K']
        width = meta['width']
        height = meta['height']
        class_ids = meta['class_ids']
        rotations = meta['rotations']
        translations = meta['translations']

        result = []
        for score, clsid, R, T in pred:
            pt3d = np.array(keypoints_3d[clsid])
            transM = np.array(
                [[width/internal_width, 0, 0],
                [0, height/internal_height, 0],
                [0, 0, 1]], dtype=np.float32)
            newR, newT, remap_err = remap_pose(internal_K, R, T, pt3d, K, transM)
            result.append([score, clsid, newR, newT, remap_err])
        
        # rearrange ground truth
        gt = []
        for i in range(len(class_ids)):
            clsid = class_ids[i]
            R = rotations[i]
            T = translations[i]
            gt.append([clsid, R, T])

        new_results.update({imagename:{'K':K, 'pred':result, 'gt':gt}})

    return new_results

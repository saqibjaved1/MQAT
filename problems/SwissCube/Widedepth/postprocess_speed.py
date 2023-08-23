import torch
from torch import nn
import cv2
import numpy as np

from loss import permute_and_flatten
from utils import (
    solve_PnP_LHM, 
    pose_symmetry_handling
)

class PostProcessor(nn.Module):
    def __init__(
        self, inference_th, box_coder, positive_num, positive_lambda, sym_types):
        super(PostProcessor, self).__init__()
        self.inference_th = inference_th
        self.positive_num = positive_num
        self.positive_lambda = positive_lambda
        self.box_coder = box_coder
        self.sym_types = sym_types

    def forward_for_single_feature_map(self, sCls, sReg, sAnchors):
        N, _, H, W = sCls.shape
        C = sReg.size(1) // 16
        A = 1

        # put in the same format as anchors
        sCls = permute_and_flatten(sCls, N, A, C, H, W)
        sCls = sCls.sigmoid()

        sReg = permute_and_flatten(sReg, N, A, C*16, H, W)
        sReg = sReg.reshape(N, -1, C*16)

        # candidate_inds = sCls >= 0.01                                                                                                                                                                                              
        candidate_inds = sCls > self.inference_th                                                                                                                                                                                   
        pre_ransac_top_n = candidate_inds.reshape(N, -1).sum(1)

        # centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        # centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores
        # sCls = sCls * centerness[:, :, None]

        results = []
        for per_cls, per_reg, per_pre_ransac_top_n, per_candidate_inds, per_anchors \
            in zip(sCls, sReg, pre_ransac_top_n, candidate_inds, sAnchors):                                                                                                                                                                                                                                                                                               
            per_cls = per_cls[per_candidate_inds]                                                                                                                                                                                    
            per_cls, top_k_indices = per_cls.topk(per_pre_ransac_top_n, sorted=False)                                                                                                                                                
            if len(per_cls) == 0:
                results.append(None)
                continue
            
            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]
            
            per_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            detections = self.box_coder.decode(
                per_reg.view(-1, C, 16)[per_loc, per_class, :],
                per_anchors.bbox[per_loc, :]
            )

            results.append([detections, per_class + 1, torch.sqrt(per_cls)])
            # boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            # boxlist.add_field("labels", per_class)
            # boxlist.add_field("scores", torch.sqrt(per_cls))
            # boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = remove_small_boxes(boxlist, self.min_size)
            # results.append(boxlist)

        return results

    def forward(self, box_cls, box_regression, targets, anchors):
        sampled_boxes = []
        anchors = list(zip(*anchors))

        # debug = True
        debug = False
        for layerIdx, (o, b, a) in enumerate(zip(box_cls, box_regression, anchors)):
            # if layerIdx != 3: #single level evaluation
            #     continue
            sampled_boxes.append(self.forward_for_single_feature_map(o, b, a))
            # debug
            if debug:
                bi = 0
                cls_scores, cls_ids = o[bi].permute(1,2,0).sigmoid().max(dim=2)
                # centerness_scores = c[bi][0].sigmoid()
                xy_predictions = b[bi].permute(1,2,0)
                nH, nW, _ = xy_predictions.shape
                xy_predictions = xy_predictions.reshape(nH*nW,-1,16)[torch.arange(0,nH*nW), cls_ids.view(-1)]
                xy_predictions = self.box_coder.decode(xy_predictions, a[bi].bbox)
                # 
                cls_scores = cls_scores.to('cpu').numpy()
                cls_ids = cls_ids.to('cpu').numpy()
                # centerness_scores = centerness_scores.to('cpu').numpy()
                xy_predictions = xy_predictions.to('cpu').numpy().reshape(-1,2,8)

                w = targets[0].width
                h = targets[0].height

                # draw xy predictions
                blackImg = np.zeros((h,w,3), np.uint8)
                # blackImg = np.ones((h,w,3), np.uint8) * 255
                xs = xy_predictions[:,0,:][(cls_scores > self.inference_th).reshape(-1)]
                ys = xy_predictions[:,1,:][(cls_scores > self.inference_th).reshape(-1)]
                xs = xs.reshape(-1)
                ys = ys.reshape(-1)
                for ki in range(len(xs)):
                    blackImg = cv2.circle(blackImg, (xs[ki], ys[ki]), radius=1, color=(255,255,255), thickness=2)
                cv2.imshow("xy predictions " + str(layerIdx), blackImg)

                cls_conf_img = (cv2.resize(cls_scores, (w,h), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
                cv2.imshow('class_conf ' + str(layerIdx), cls_conf_img)

                cls_max_id_img = cv2.resize(cls_ids, (w,h), interpolation=cv2.INTER_NEAREST)
                cls_max_id_img = cv2.normalize(cls_max_id_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow('class_id ' + str(layerIdx), cls_max_id_img)

                # centerness_img = (cv2.resize(centerness_scores, (w,h), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
                # cv2.imshow('centerness ' + str(layerIdx), centerness_img)
        if debug:
            cv2.waitKey(0)

        pred_inter_list = list(zip(*sampled_boxes))
        return self.select_over_all_levels(pred_inter_list, targets)

        # boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        # if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
        #     boxlists = self.select_over_all_levels(boxlists)

        # return boxlists

    def select_over_all_levels(self, pred_inter_list, targets):
        num_images = len(pred_inter_list)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = self.pose_infer_ml(pred_inter_list[i], targets[i])

            # symmetry handling
            if self.sym_types is not None and len(self.sym_types) > 0:
                for j in range(len(result)):
                    score, cls_id, R, T = result[j]
                    key_str = "cls_" + str(cls_id)
                    if key_str in self.sym_types:
                        R = pose_symmetry_handling(R, self.sym_types[key_str])
                        result[j] = score, cls_id, R, T

            # number_of_detections = len(result)
            # # Limit to max_per_image detections **over all classes**
            # if number_of_detections > self.fpn_post_nms_top_n > 0:
            #     cls_scores = result.get_field("scores")
            #     image_thresh, _ = torch.kthvalue(
            #         cls_scores.cpu(),
            #         number_of_detections - self.fpn_post_nms_top_n + 1
            #     )
            #     keep = cls_scores >= image_thresh.item()
            #     keep = torch.nonzero(keep).squeeze(1)
            #     result = result[keep]
            results.append(result)
        return results

    def pose_infer_ml(self, preds, target):
        K = target.K
        keypoints_3d = target.keypoints_3d
        # 
        # extract valid preds from multiple layers
        preds_mgd = [p for p in preds if p is not None]
        if len(preds_mgd) == 0:
            return []
        # merge labels from multi layers
        _, labels, _ = list(zip(*preds_mgd))
        candi_labels = torch.unique(torch.cat(labels, dim=0))
        # 
        results = []
        for lb in candi_labels:
            clsId = lb - 1
            # 
            # fetch only desired cells
            # 
            validCntPerLayer = [0]*len(preds)
            # get the reprojected box size of maximum confidence
            boxSize = 0
            boxConf = 0
            detects = [[]] * len(preds)
            scores = [[]] * len(preds)
            for i in range(len(preds)):
                item = preds[i]
                if item is not None: 
                    det, lbl, scs = item
                    mask = (lbl == lb) # choose the current label only
                    det = det[mask]
                    scs = scs[mask]
                    detects[i] = det
                    scores[i] = scs
                    # 
                    validCntPerLayer[i] = len(scs)
                    if len(scs) > 0:
                        idx = torch.argmax(scs)
                        if scs[idx] > boxConf:
                            boxConf = scs[idx]
                            kpts = det[idx].view(2, -1)
                            size = max(kpts[0].max()-kpts[0].min(), kpts[1].max()-kpts[1].min())
                            if size > boxSize:
                                boxSize = size
            # compute the desired cell numbers for each layer
            dk = torch.log2(boxSize / torch.FloatTensor(self.box_coder.anchor_sizes).type_as(boxSize))
            nk = torch.exp(-self.positive_lambda * (dk * dk))
            nk = self.positive_num * nk / nk.sum(0, keepdim=True)
            nk = (nk + 0.5).int()

            # extract most confident cells
            detection_per_lb = []
            scores_per_lb = []
            for i in range(len(preds)):
                pkNum = min(validCntPerLayer[i], nk[i])
                if pkNum > 0:
                    scs, indexes = scores[i].topk(pkNum)
                    detection_per_lb.append(detects[i][indexes])
                    scores_per_lb.append(scs)
            if len(scores_per_lb) == 0:
                continue
            detection_per_lb = torch.cat(detection_per_lb)
            scores_per_lb = torch.cat(scores_per_lb)

            # PnP solver
            xy3d = keypoints_3d[clsId].repeat(len(scores_per_lb), 1, 1)
            xy2d = detection_per_lb.view(len(scores_per_lb), 2, -1).transpose(1, 2).contiguous()
            
            # CPU is more effective here
            K = K.to('cpu')
            xy3d = xy3d.to('cpu')
            xy2d = xy2d.to('cpu')

            if False:
                R, T, errs = solve_PnP_LHM(K, xy3d.view(-1, 3), xy2d.view(-1, 2))
                R = R.numpy()
                T = T.numpy()
                results.append([float(scores_per_lb.max()), int(clsId), R, T])
            else:
                xy3d_np = xy3d.view(-1,3).numpy()
                xy2d_np = xy2d.view(-1,2).numpy()
                K_np = K.numpy()

                retval, rot, trans, inliers = cv2.solvePnPRansac(xy3d_np, xy2d_np, K_np, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0)
                # retval, rot, trans = cv2.solvePnP(tmpv, xy2d, rawK, None, flags=cv2.SOLVEPNP_ITERATIVE)

                if retval:
                    # print('%d/%d' % (len(inliers), len(xy2d_np)))
                    R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
                    T = trans.reshape(-1, 1)

                    if np.isnan(R.sum()) or np.isnan(T.sum()):
                        continue

                    results.append([float(scores_per_lb.max()), int(clsId), R, T])
        return results

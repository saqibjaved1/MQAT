import torch
from torch import nn
import cv2
import numpy as np

from .loss import permute_and_flatten

class PostProcessor(nn.Module):
    def __init__(
        self, inference_th, num_classes, box_coder, positive_num, positive_lambda):
        super(PostProcessor, self).__init__()
        self.inference_th = inference_th
        self.num_classes = num_classes
        self.positive_num = positive_num
        self.positive_lambda = positive_lambda
        self.box_coder = box_coder

    def forward_for_single_feature_map(self, sCls, sReg, sAnchors):
        N, _, H, W = sCls.shape
        C = sReg.size(1) // 16
        A = 1

        # put in the same format as anchors
        sCls = permute_and_flatten(sCls, N, A, C, H, W)
        sCls = sCls.sigmoid()

        sReg = permute_and_flatten(sReg, N, A, C*16, H, W)
        sReg = sReg.reshape(N, -1, C*16)

        candidate_inds = sCls > self.inference_th
        # print(candidate_inds)

        pre_ransac_top_n = candidate_inds.view(N, -1).sum(1)
        # print(pre_ransac_top_n)
        # exit()
        results = []
        for per_sCls, per_sReg, per_pre_ransac_top_n, per_candidate_inds, per_anchors \
                in zip(sCls, sReg, pre_ransac_top_n, candidate_inds, sAnchors):

            per_sCls = per_sCls[per_candidate_inds]
            per_sCls, top_k_indices = per_sCls.topk(per_pre_ransac_top_n, sorted=False)
            if len(per_sCls) == 0:
                results.append(None)
                continue

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            detections = self.box_coder.decode(
                per_sReg.view(-1, C, 16)[per_box_loc, per_class],
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            results.append([detections, per_class + 1, torch.sqrt(per_sCls)])

        return results

    def forward(self, pred_cls, pred_reg, pred_cls2, pred_reg2, pred_cls3, pred_reg3, pred_cls4, pred_reg4, targets, anchors):
        #before 4 heads
        # sampled_boxes = []
        # anchors = list(zip(*anchors))
        # for layerIdx, (o, b, a) in enumerate(zip(pred_cls, pred_reg, anchors)):
        #     sampled_boxes.append(self.forward_for_single_feature_map(o, b, a))
        # pred_inter_list = list(zip(*sampled_boxes))
        # return self.select_over_all_levels(pred_inter_list, targets)

        sampled_boxes = []
        sampled_boxes2 = []
        sampled_boxes3 = []
        sampled_boxes4 = []
        # anchors = 2*anchors
        # print(len(anchors),len(pred_cls))
        anchors = list(zip(*anchors))
        results = []
        for layerIdx, (o, b, a) in enumerate(zip(pred_cls, pred_reg, anchors)):
            # print(layerIdx)
            sampled_boxes.append(self.forward_for_single_feature_map(o, b, a))
        pred_inter_list = list(zip(*sampled_boxes))

        for layerIdx, (o, b, a) in enumerate(zip(pred_cls2, pred_reg2, anchors)):
            # print(layerIdx)
            sampled_boxes2.append(self.forward_for_single_feature_map(o, b, a))
        pred_inter_list2 = list(zip(*sampled_boxes2))

        for layerIdx, (o, b, a) in enumerate(zip(pred_cls3, pred_reg3, anchors)):
            # print(layerIdx)
            sampled_boxes3.append(self.forward_for_single_feature_map(o, b, a))
        pred_inter_list3 = list(zip(*sampled_boxes3))

        for layerIdx, (o, b, a) in enumerate(zip(pred_cls4, pred_reg4, anchors)):
            # print(layerIdx)
            sampled_boxes4.append(self.forward_for_single_feature_map(o, b, a))
        pred_inter_list4 = list(zip(*sampled_boxes4))

        # num_images = len(pred_inter_list)
        # for i in range(num_images):
        return self.select_over_all_levels(pred_inter_list, pred_inter_list2, pred_inter_list3, pred_inter_list4, targets)

        # return results

    def select_over_all_levels(self, pred_inter_list, pred_inter_list2, pred_inter_list3, pred_inter_list4, targets):
        num_images = len(pred_inter_list)
        # print("no. of images", num_images)
        results = []
        # print((pred_inter_list))
        for i in range(num_images):
            result = self.pose_infer_ml(pred_inter_list[i], pred_inter_list2[i], pred_inter_list3[i], pred_inter_list4[i], targets[i])
            results.append(result)
        return results

    def pose_infer_ml(self, preds1, preds2, preds3, preds4, target):
        K = target.K
        keypoints_3d = target.keypoints_3d
        # 
        # extract valid preds from multiple layers
        preds_mgd1 = [p for p in preds1 if p is not None]
        preds_mgd2 = [p for p in preds2 if p is not None]
        preds_mgd3 = [p for p in preds3 if p is not None]
        preds_mgd4 = [p for p in preds4 if p is not None]
        # print(len(preds_mgd3), len(preds_mgd2), len(preds_mgd), len(preds_mgd2))
        # print(len(preds_mgd+preds_mgd2))
        # print(preds_mgd1,preds_mgd2,preds_mgd+preds_mgd2)
        preds_mgd_ens = preds_mgd1+ preds_mgd2 + preds_mgd3 + preds_mgd4

        preds_list = preds1 + preds2 + preds3 + preds4

        preds_mgd_2 = [p for p in preds_list if p is not None]

        # print(preds_mgd_ens==preds_mgd_2)

        if len(preds_mgd1)+len(preds_mgd2)+len(preds_mgd3)+len(preds_mgd4) == 0:
            return []
        # merge labels from multi layers
        _, labels, _ = list(zip(*preds_mgd_ens))
        candi_labels = torch.unique(torch.cat(labels, dim=0))
        # print(candi_labels.item())
        #
        results = []

        preds_cat = [preds1,preds2,preds3,preds4]

        for lb in candi_labels:

            detection_per_lb = []
            scores_per_lb = []
            c=0

            for preds in preds_cat:
                # c = c + 1
                # print(c)
                clsId = lb - 1
                #
                # fetch only desired cells
                #
                validCntPerLayer = [0]*len(preds)
                # get the reprojected box size of maximum confidence
                boxSize = 0.
                boxConf = 0
                detects = [[]] * len(preds)
                scores = [[]] * len(preds)
                for i in range(len(preds)):
                    item = preds[i]
                    # print(preds[i])
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

                # validCntPerLayer = [0] * len(preds)
                # # get the reprojected box size of maximum confidence
                # boxSize = 0
                # boxConf = 0
                # detects = [[]] * len(preds)
                # scores = [[]] * len(preds)
                # for i in range(len(preds)):
                #     item = preds[i]
                #     # print(preds[i])
                #     if item is not None:
                #         det, lbl, scs = item
                #         mask = (lbl == lb)  # choose the current label only
                #         det = det[mask]
                #         scs = scs[mask]
                #         detects[i] = det
                #         scores[i] = scs
                #         #
                #         validCntPerLayer[i] = len(scs)
                #         if len(scs) > 0:
                #             idx = torch.argmax(scs)
                #             if scs[idx] > boxConf:
                #                 boxConf = scs[idx]
                #                 kpts = det[idx].view(2, -1)
                #                 size = max(kpts[0].max() - kpts[0].min(), kpts[1].max() - kpts[1].min())
                #                 if size > boxSize:
                #                     boxSize = size
                #
                # validCntPerLayer = [0] * len(preds)
                # # get the reprojected box size of maximum confidence
                # boxSize = 0
                # boxConf = 0
                # detects = [[]] * len(preds)
                # scores = [[]] * len(preds)
                # for i in range(len(preds)):
                #     item = preds[i]
                #     # print(preds[i])
                #     if item is not None:
                #         det, lbl, scs = item
                #         mask = (lbl == lb)  # choose the current label only
                #         det = det[mask]
                #         scs = scs[mask]
                #         detects[i] = det
                #         scores[i] = scs
                #         #
                #         validCntPerLayer[i] = len(scs)
                #         if len(scs) > 0:
                #             idx = torch.argmax(scs)
                #             if scs[idx] > boxConf:
                #                 boxConf = scs[idx]
                #                 kpts = det[idx].view(2, -1)
                #                 size = max(kpts[0].max() - kpts[0].min(), kpts[1].max() - kpts[1].min())
                #                 if size > boxSize:
                #                     boxSize = size
                #
                # validCntPerLayer = [0] * len(preds)
                # # get the reprojected box size of maximum confidence
                # boxSize = 0
                # boxConf = 0
                # detects = [[]] * len(preds)
                # scores = [[]] * len(preds)
                # for i in range(len(preds)):
                #     item = preds[i]
                #     # print(preds[i])
                #     if item is not None:
                #         det, lbl, scs = item
                #         mask = (lbl == lb)  # choose the current label only
                #         det = det[mask]
                #         scs = scs[mask]
                #         detects[i] = det
                #         scores[i] = scs
                #         #
                #         validCntPerLayer[i] = len(scs)
                #         if len(scs) > 0:
                #             idx = torch.argmax(scs)
                #             if scs[idx] > boxConf:
                #                 boxConf = scs[idx]
                #                 kpts = det[idx].view(2, -1)
                #                 size = max(kpts[0].max() - kpts[0].min(), kpts[1].max() - kpts[1].min())
                #                 if size > boxSize:
                #                     boxSize = size
                #
                #
                #
                # compute the desired cell numbers for each layer
                # if boxConf==0:
                #     print("boxsize=0")
                #     continue
                if boxSize==0:
                    boxSize=torch.tensor(boxSize)
                dk = torch.log2(boxSize / torch.FloatTensor(self.box_coder.anchor_sizes).type_as(boxSize))
                nk = torch.exp(-self.positive_lambda * (dk * dk))
                nk = self.positive_num * nk / nk.sum(0, keepdim=True)
                nk = (nk + 0.5).int()


                # extract most confident cells
                detection_per_lb_per_head = []
                scores_per_lb_per_head = []
                for i in range(len(preds)):
                    pkNum = min(validCntPerLayer[i], nk[i])
                    if pkNum > 0:
                        scs, indexes = scores[i].topk(pkNum)
                        detection_per_lb_per_head.append(detects[i][indexes])
                        scores_per_lb_per_head.append(scs)
            # for i in range(len(preds2)):
            #     pkNum = min(validCntPerLayer[i], nk[i])
            #     if pkNum > 0:
            #         scs, indexes = scores[i].topk(pkNum)
            #         detection_per_lb.append(detects[i][indexes])
            #         scores_per_lb.append(scs)
            # for i in range(len(preds3)):
            #     pkNum = min(validCntPerLayer[i], nk[i])
            #     if pkNum > 0:
            #         scs, indexes = scores[i].topk(pkNum)
            #         detection_per_lb.append(detects[i][indexes])
            #         scores_per_lb.append(scs)
            # for i in range(len(preds4)):
            #     pkNum = min(validCntPerLayer[i], nk[i])
            #     if pkNum > 0:
            #         scs, indexes = scores[i].topk(pkNum)
            #         detection_per_lb.append(detects[i][indexes])
            #         scores_per_lb.append(scs)
            #
            # if len(scores_per_lb) == 0:
            #     continue
                if len(scores_per_lb_per_head) == 0:
                    continue
                detection_per_lb = torch.cat(detection_per_lb_per_head)
                scores_per_lb = torch.cat(scores_per_lb_per_head)

                # PnP solver
            if len(scores_per_lb) == 0:
                continue
                # print(detection_per_lb)
            xy3d = keypoints_3d[clsId].repeat(len(scores_per_lb), 1, 1)
            xy2d = detection_per_lb.view(len(scores_per_lb), 2, -1).transpose(1, 2).contiguous()

            # CPU is more effective here
            K = K.to('cpu')
            xy3d = xy3d.to('cpu')
            xy2d = xy2d.to('cpu')

            xy3d_np = xy3d.view(-1,3).numpy()
            xy2d_np = xy2d.view(-1,2).detach().numpy()
            K_np = K.numpy()

            retval, rot, trans, inliers = cv2.solvePnPRansac(xy3d_np, xy2d_np, K_np, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0)

            if retval:
                # print('%d/%d' % (len(inliers), len(xy2d_np)))
                R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
                T = trans.reshape(-1, 1)

                if np.isnan(R.sum()) or np.isnan(T.sum()):
                    continue

                results.append([float(scores_per_lb.max()), int(clsId), R, T])
        return results

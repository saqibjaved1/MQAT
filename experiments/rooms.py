import torch
import horovod.torch as hvd
import cv2
import os
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
from torch import nn
from problems.SwissCube.Widedepth.distributed import (
    get_rank,
    reduce_loss_dict,
)

from problems.SwissCube.Widedepth.argument import get_args

from problems.SwissCube.Widedepth.utils import (
    load_bop_meshes,
    visualize_pred,
    print_accuracy_per_class,
    accumulate_dicts
)
from problems.SwissCube.Widedepth.evaluate import evaluate, evaluate_pose_predictions

def train(logbook, train_l, net, loss_fn, opt, ctrls):
    """Run one epoch of the training experiment."""
    net.train()
    for c in ctrls:
        c.step_pre_training(logbook.i_epoch, opt, logbook)
    #hvd.broadcast_parameters(net.state_dict(), root_rank=logbook.sw_cfg['master_rank'])
    count=0
    logbook.meter.reset()
    for i_batch, data in enumerate(train_l):
        opt.zero_grad()
        # load data to device
        inputs, gt_labels = data[0],data[1]
        
        inputs            = inputs.to(logbook.hw_cfg['device'])
        gt_labels         = gt_labels.to(logbook.hw_cfg['device'])
        # forprop
        if logbook.config["experiment"]["task"]=="classification":
            pr_outs       = net(inputs)
            loss          = loss_fn(pr_outs, gt_labels)
            # backprop
            loss.backward()
            opt.step()
               # update statistics
            logbook.meter.update(pr_outs, gt_labels, loss)

            if logbook.verbose:
                print('Training\t [{:>4}/{:>4}]'.format(logbook.i_epoch+1, logbook.config['experiment']['n_epochs']), end='')
                print(' | Batch [{:>5}/{:>5}]'.format(i_batch+1, len(train_l)), end='')
                print(' | Loss: {:6.3f} - Metric: {:6.2f}'.format(logbook.meter.avg_loss, logbook.meter.avg_metric))

            # log statistics to file
            stats = {
                'train_loss':   logbook.meter.avg_loss,
                'train_metric': logbook.meter.avg_metric
            }
        else:
            loss_dict     = net(inputs,targets=gt_labels)
            loss          = sum(loss for loss in loss_dict.values())
                # backprop
            loss.backward()
            opt.step()
            # update statistics
            logbook.meter.update(total_loss=loss,**loss_dict)
        
            if logbook.verbose:
                print('Training\t [{:>4}/{:>4}]'.format(logbook.i_epoch+1, logbook.config['experiment']['n_epochs']), end='')
                print(' | Batch [{:>5}/{:>5}]'.format(i_batch+1, len(train_l)), end='')
                for name, meter in logbook.meter.meters.items():
                    
                    print(" ", name,": ", (meter.avg))

                print('Learning rate: ', opt.param_groups[0]['lr'])

            # log statistics to file
            stats = {
                'train_loss':   logbook.meter.meters["total_loss"].avg
                #'train_metric': logbook.meter.avg_metric
            }
     
        
        
    if logbook.is_master:
        for k, v in stats.items():
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
        logbook.writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], global_step=logbook.i_epoch)

    return stats


def validate(logbook, valid_l, net, loss_fn, ctrls):
    """Run a validation epoch."""
    net.eval()
    for c in ctrls:
        c.step_pre_validation(logbook.i_epoch)

    with torch.no_grad():
        logbook.meter.reset()
        for i_batch, data in enumerate(valid_l):
            # load data to device
            inputs, gt_labels = data
            inputs            = inputs.to(logbook.hw_cfg['device'])
            gt_labels         = gt_labels.to(logbook.hw_cfg['device'])
            # forprop
            pr_outs           = net(inputs)
            loss              = loss_fn(pr_outs, gt_labels)
            # update statistics
            logbook.meter.update(pr_outs, gt_labels, loss)

            if logbook.verbose:
                print('Validation\t [{:>4}/{:>4}]'.format(logbook.i_epoch+1, logbook.config['experiment']['n_epochs']), end='')
                print(' | Batch [{:>5}/{:>5}]'.format(i_batch+1, len(valid_l)), end='')
                print(' | Loss: {:6.3f} - Metric: {:6.2f}'.format(logbook.meter.avg_loss, logbook.meter.avg_metric))

    # log statistics to file
    stats = {
        'valid_loss':   logbook.meter.avg_loss,
        'valid_metric': logbook.meter.avg_metric
    }
    if logbook.is_master:
        for k, v in stats.items():
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)

    return stats


def train_6D(loader, model, optimizer, logbook,ctrls,trackers):
    model.train()

    for c in ctrls:
        c.step_pre_training(logbook.i_epoch, optimizer, logbook)
    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)
    count=0
    for idx, (images, targets, _) in pbar:
        # count += 1
        # if count == 10:
        #     break

        # for t in trackers:
        #     t.setup(model, logbook.i_epoch+1)

        model.zero_grad()

        images = images.to(logbook.hw_cfg['device'])
        targets = [target.to(logbook.hw_cfg['device']) for target in targets]

        _, loss_dict = model(images, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_reg = loss_dict['loss_reg'].mean()
        # loss_cls2 = loss_dict['loss_cls2'].mean()
        # loss_reg2 = loss_dict['loss_reg2'].mean()
        # loss_cls3 = loss_dict['loss_cls3'].mean()
        # loss_reg3 = loss_dict['loss_reg3'].mean()
        # loss_cls4 = loss_dict['loss_cls4'].mean()
        # loss_reg4 = loss_dict['loss_reg4'].mean()

        # loss = loss_cls + loss_reg + loss_cls2 + loss_reg2 + loss_cls3 + loss_reg3 + loss_cls4 + loss_reg4
        loss = loss_cls + loss_reg

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # for t in trackers:
        #     t.release()
        # ckpt2 = torch.load("/cvlabdata2/home/javed/quantlab/problems/SwissCube/swisscube_pretrained.pth")
        # dic = model.state_dict()
        # print(dic["model.head.pose_tower.7.bias"] == ckpt2["head.pose_tower.7.bias"])
        # loss_reduced = reduce_loss_dict(loss_dict)
        # loss_cls = loss_reduced['loss_cls'].mean().item()
        # loss_reg = loss_reduced['loss_reg'].mean().item()

        if get_rank() == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar_str = (("epoch: %d/%d, lr:%.6f, cls:%.4f, reg:%.4f") % (logbook.i_epoch+1, logbook.config['experiment']['n_epochs'], current_lr, loss_cls, loss_reg))
            pbar.set_description(pbar_str)

            # writing log to tensorboard
            # if logger and idx % 10 == 0:
            #     # totalStep = (epoch * len(loader) + idx) * args.batch * args.n_gpu
            #     totalStep = (logbook.i_epoch * len(loader) + idx) * cfg['SOLVER']['IMS_PER_BATCH']
        # logbook.writer.add_scalar('training/learning_rate', current_lr, logbook.i_epoch)
        # logbook.writer.add_scalar('training/loss_cls', loss_cls, logbook.i_epoch)
        # logbook.writer.add_scalar('training/loss_reg', loss_reg, logbook.i_epoch)
        # logbook.writer.add_scalar('training/loss_all', (loss_cls + loss_reg), logbook.i_epoch)
    return loss_cls, loss_reg

def valid_6D(loader, model, logbook):
    torch.cuda.empty_cache()
    cfg = get_args()
    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    preds = {}

    meshes, _ = load_bop_meshes(cfg['DATASETS']['MESH_DIR'])
    count = 0
    for idx, (images, targets, meta_infos) in pbar:

        # count +=1
        # if count==20:
        #     break
        model.zero_grad()

        images = images.to(logbook.hw_cfg['device'])
        targets = [target.to(logbook.hw_cfg['device']) for target in targets]

        pred = model(images, targets=targets)

        # if get_rank() == 0 and idx % 10 == 0:
        #     bIdx = 0
        #     imgpath, imgname = os.path.split(meta_infos[bIdx]['path'])
        #     name_prefix = imgpath.replace(os.sep, '_').replace('.', '') + '_' + os.path.splitext(imgname)[0]
        #
        #     rawImg, visImg, gtImg = visualize_pred(images.tensors[bIdx], targets[bIdx], pred[bIdx],
        #                                            cfg['INPUT']['PIXEL_MEAN'], cfg['INPUT']['PIXEL_STD'],cfg['DATASETS']['SYMMETRY_TYPES'], meshes)
        #     # cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '.png', rawImg)
        #     cv2.imwrite("/cvlabdata2/home/javed/quant-swisscube-only/WDR" + name_prefix + '_pred.png', visImg)
        #     cv2.imwrite("/cvlabdata2/home/javed/quant-swisscube-only/WDR" + name_prefix + '_gt.png', gtImg)
        #
        #     ### Draw all prediction points. (in greeen)
        #     for i in range(len(xy2d_nps)):
        #         for j in range(len(xy2d_nps[i])):
        #             pt = (int(xy2d_nps[i][j][0]), int(xy2d_nps[i][j][1]))
        #             points_Img = cv2.circle(img=gtImg, center=pt, radius=2, color=[0, 0, 255], thickness=-1)
        #     cv2.imwrite("/cvlabdata2/home/javed/quant-swisscube-only/" + name_prefix + '_points.png', points_Img)

        # pred = [p.to('cpu') for p in pred]

        for m, p in zip(meta_infos, pred):
            preds.update({m['path']: {
                'meta': m,
                'pred': p
            }})

    preds = accumulate_dicts(preds)

    if get_rank() != 0:
        return

    # accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
    #     = evaluate(logbook.config, preds)
    accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
        = evaluate_pose_predictions(preds, cfg['DATASETS']['N_CLASS'], meshes, cfg['DATASETS']['MESH_DIAMETERS'], cfg['DATASETS']['SYMMETRY_TYPES'])

    print_accuracy_per_class(accuracy_adi_per_class, accuracy_rep_per_class)

    # writing log to tensorboard
    if logbook.writer:
        classNum = cfg['DATASETS']['N_CLASS'] - 1  # get rid of background class
        print(classNum, len(accuracy_adi_per_class))
        assert (len(accuracy_adi_per_class) == classNum)
        assert (len(accuracy_rep_per_class) == classNum)

        all_adi = {}
        all_rep = {}
        validClassNum = 0

        for i in range(classNum):
            className = ('class_%02d' % i)
            logbook.writer.add_scalars('ADI/' + className, accuracy_adi_per_class[i], logbook.i_epoch)
            logbook.writer.add_scalars('REP/' + className, accuracy_rep_per_class[i], logbook.i_epoch)
            #
            assert (len(accuracy_adi_per_class[i]) == len(accuracy_rep_per_class[i]))
            if len(accuracy_adi_per_class[i]) > 0:
                for key, val in accuracy_adi_per_class[i].items():
                    if key in all_adi:
                        all_adi[key] += val
                    else:
                        all_adi[key] = val
                for key, val in accuracy_rep_per_class[i].items():
                    if key in all_rep:
                        all_rep[key] += val
                    else:
                        all_rep[key] = val
                validClassNum += 1

        # averaging
        for key, val in all_adi.items():
            all_adi[key] = val / validClassNum
        for key, val in all_rep.items():
            all_rep[key] = val / validClassNum
        logbook.writer.add_scalars('ADI/all_class', all_adi, logbook.i_epoch)
        logbook.writer.add_scalars('REP/all_class', all_rep, logbook.i_epoch)

    return accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range


def train_obj_coco():
    {

    }

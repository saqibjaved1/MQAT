import torch
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd
import quantlab.monitor as qm
import quantlab.graphs as qg

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import utils.lr_schedulers as lr_schedulers
from collections import OrderedDict

def get_data(logbook):
    """Return data for the experiment."""

    # create data sets
    train_set, valid_set = logbook.lib.load_data_sets(logbook)
    # is cross-validation experiment?
    if logbook.config["experiment"]["task"]=="classification":   
        if logbook.config['experiment']['n_folds'] > 1:
            import itertools
            torch.manual_seed(logbook.config['experiment']['seed'])  # make data set random split consistent
            indices = torch.randperm(len(train_set)).tolist()
            folds_indices = []
            for k in range(logbook.config['experiment']['n_folds']):
                folds_indices.append(indices[k::logbook.config['experiment']['n_folds']])
            train_fold_indices = list(itertools.chain(*[folds_indices[i] for i in range(len(folds_indices)) if i != logbook.i_fold]))
            valid_fold_indices = folds_indices[logbook.i_fold]
            valid_set = torch.utils.data.Subset(train_set, valid_fold_indices)
            train_set = torch.utils.data.Subset(train_set, train_fold_indices)  # overwriting `train_set` must be done in right order!

        # create samplers (maybe distributed)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=logbook.sw_cfg['global_size'], rank=logbook.sw_cfg['global_rank'])
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, num_replicas=logbook.sw_cfg['global_size'], rank=logbook.sw_cfg['global_rank'])

        # wrap data sets into loaders
        bs_train = logbook.config['data']['bs_train']
        bs_valid = logbook.config['data']['bs_valid']
        kwargs = {'num_workers': 8, 'pin_memory': True} if logbook.hw_cfg['n_gpus'] else {'num_workers': 1}
        if hasattr(train_set, 'collate_fn'):  # if one data set needs `collate`, all the data sets should need it
            train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, sampler=train_sampler, collate_fn=train_set.collate_fn, **kwargs)
            valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, sampler=valid_sampler, collate_fn=valid_set.collate_fn, **kwargs)
        else:
            train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, sampler=train_sampler, **kwargs)
            valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, sampler=valid_sampler, **kwargs)
        return train_l,valid_l
      
    return train_set, valid_set


def get_network(logbook):
    """Return a network for the experiment and the loss function for training."""

    # create the network
    net = getattr(logbook.lib, logbook.config['network']['class'])(**logbook.config['network']['params'])

    # Compute and print the model size
    total_params_count = sum(p.numel() for p in net.parameters())
    print("Model size of the model: %d parameters" % total_params_count)
    total_params_count = sum(p.numel() for p in net.backbone.parameters())
    print("Model size of the backbone model: %d parameters" % total_params_count)
    total_params_count = sum(p.numel() for p in net.model.fpn.parameters())
    print("Model size of the fpn: %d parameters" % total_params_count)
    total_params_count = sum(p.numel() for p in net.model.head.parameters())
    print("Model size of the head: %d parameters" % total_params_count)

    # quantize (if specified)
    # changed places for quantization
    if logbook.config['network']['quantize'] is not None:
        quant_convert = getattr(logbook.lib, logbook.config['network']['quantize']['routine'])
        net = quant_convert(logbook.config['network']['quantize'], net)


    if logbook.config['experiment']['pretrained']:
        checkpoint = torch.load(logbook.config['experiment']['url'])

        network_dict = net.state_dict()
        dict = OrderedDict()
        if logbook.config["experiment"]["task"]=="classification":
            for k, v in network_dict.items():
                if k in checkpoint.keys():
                    dict[k] = checkpoint[k]
                elif k.find("conv2.conv") != -1 and k.find("frozen") == -1 and k.find("weight_s") == -1:
                    dict[k]=checkpoint[k.replace("conv2.conv","conv2")]
                elif k.find("fc.conv") != -1 and k.find("frozen") == -1 and k.find("weight_s") == -1:
                    dict[k] = checkpoint[k.replace("fc.conv", "fc")]
                elif k.find("conv1.conv") != -1 and k.find("frozen") == -1 and k.find("weight_s") == -1:
                    dict[k] = checkpoint[k.replace("conv1.conv", "conv1")]
                else:
                    dict[k]= network_dict[k]
            net.load_state_dict(dict)

        elif logbook.config["experiment"]["task"]=="6D":

            for k, v in network_dict.items():

                if k.find("head2") != -1:
                    dict[k]= network_dict[k.replace("head2", "head")]
                elif k.find("head3") != -1:
                    dict[k] = network_dict[k.replace("head3", "head")]
                elif k.find("head4") != -1:
                    dict[k]= network_dict[k.replace("head4", "head")]

                elif k.find("weight_s") != -1:
                    if k.find("model") != -1:
                        dict[k] = nn.Parameter(
                            torch.full((1,), float('NaN')).to(checkpoint[k.replace("weight_s", "weight").replace("model.","")]),
                            requires_grad=False)
                        # dict["model." + k] = nn.Parameter(
                        #     torch.full((1,), float('NaN')).to(checkpoint[k.replace("weight_s", "weight").replace("model.","")]),
                        #     requires_grad=False)
                    else:
                        dict[k] = nn.Parameter(
                            torch.full((1,), float('NaN')).to(checkpoint[k.replace("weight_s", "weight")]),
                            requires_grad=False)
                        # dict["model." + k] = nn.Parameter(
                        #     torch.full((1,), float('NaN')).to(checkpoint[k.replace("weight_s", "weight")]),
                        #     requires_grad=False)

                elif k.find("weight_frozen") != -1:
                    if k.find("model")!=-1:
                        #dict[k] = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight").replace("model.","")].data.,requires_grad=False)
                        # dict["model." + k] = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight").replace("model.","")].data,
                        #                                   requires_grad=False)
                        # size = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight").replace("model.","")].data,requires_grad=False).shape
                        # dict[k]= torch.full(size,float('NaN'))
                        dict[k] = network_dict[k]
                    else:
                        #dict[k] = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight")].data,requires_grad=False)
                        # dict["model." + k] = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight")].data,
                        #                     requires_grad=False)
                        # size = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight")].data,requires_grad=False).shape
                        # dict[k] = torch.full(size, float('NaN'))
                        dict[k] = network_dict[k]

                elif k.find("fpn") != -1 or k.find("head") != -1 or k.find("anchors") != -1 or k.find("head2") != -1:
                    dict[k] = checkpoint[k.replace("model.", "")]

                if k in checkpoint.keys():

                    if k.find("backbone") != -1:
                        dict[k] = checkpoint[k]
                        dict["model." + k] = checkpoint[k]

                    dict["model." + k] = checkpoint[k]

            # dict = OrderedDict()
            # for k, v in network_dict.items():
            #
            #     # if k.find("weight_s") != -1 and k.find("backbone")!=-1:
            #     if k.find("weight_s") != -1 and (k.find("backbone")!=-1 or k.find("head")!=-1):
            #         dict[k] = nn.Parameter(
            #             torch.full((1,), float('NaN')).to(checkpoint[k.replace("weight_s", "weight")]),
            #             requires_grad=False)
            #
            #     elif k.find("weight_frozen") != -1 and (k.find("backbone")!=-1 or k.find("head")!=-1):
            #         # dict[k] = nn.Parameter(checkpoint[k.replace("weight_frozen", "weight").replace("model.","")].data,requires_grad=False)
            #         dict[k] = nn.Parameter(
            #             torch.full_like(checkpoint[k.replace("weight_frozen", "weight")].data, float('NaN')),
            #             requires_grad=False)
            #     else:
            #         dict[k] = checkpoint[k]
            # #
            net.load_state_dict(dict)


        else:
            net.load_state_dict(checkpoint)

    # if logbook.config['network']['quantize'] is not None:
    #     quant_convert = getattr(logbook.lib, logbook.config['network']['quantize']['routine'])
    #     net = quant_convert(logbook.config['network']['quantize'], net)
    # move to proper device
    net = net.to(logbook.hw_cfg['device'])

    # for l in qg.analyse.list_nodes(net):
    #     print(l.name)

    # print(net.state_dict()["model.head.pose_tower.10.bias"])

    return net


def get_training(logbook, net):
    """Return a training procedure for the experiment."""
 
    # optimization algorithm
    opt_choice = {**optim.__dict__}
    logbook.config['training']['optimizer']['params']['lr'] *= logbook.sw_cfg['global_size']  # adjust learning rate
    opt        = opt_choice[logbook.config['training']['optimizer']['class']](net.parameters(), **logbook.config['training']['optimizer']['params'])
    opt        = hvd.DistributedOptimizer(opt, named_parameters=net.named_parameters())

    # learning rate scheduler
    lr_sched_choice = {**optim.lr_scheduler.__dict__, **lr_schedulers.__dict__}
    lr_sched        = lr_sched_choice[logbook.config['training']['lr_scheduler']['class']](opt, **logbook.config['training']['lr_scheduler']['params'])

    # quantization controllers (if specified)
    if logbook.config['training']['quantize']:
        quant_controls = getattr(logbook.lib, logbook.config['training']['quantize']['routine'])
        ctrls = quant_controls(logbook.config['training']['quantize'], net)
    else:
        ctrls = []
    
    if logbook.config["experiment"]["task"]=="classification":
        # loss function
        loss_fn_choice = {**nn.__dict__, **logbook.lib.__dict__}
        loss_fn_class  = loss_fn_choice[logbook.config['training']['loss_function']['class']]
        if 'net' in loss_fn_class.__init__.__code__.co_varnames:
            loss_fn = loss_fn_class(net, **logbook.config['training']['loss_function']['params'])
        else:
            loss_fn = loss_fn_class(**logbook.config['training']['loss_function']['params'])

        return loss_fn,opt,lr_sched,ctrls

    return "", opt, lr_sched, ctrls


def analyze_net(logbook,net):
    """Logs Sparsity for each layer"""

    total_zeros, params = 0, 0

    for name, param in net.named_parameters():
        if name.find("frozen") != -1:
            zeros_layer = (param.numel() - param.nonzero().size(0))
            total_zeros += zeros_layer
            params += param.numel()
            # if logbook.is_master:
            #     logbook.writer.add_scalar(name, zeros_layer/param.numel(), global_step=logbook.i_epoch)
    if logbook.is_master:
        logbook.writer.add_scalar("total_sparsity", total_zeros/params, global_step=logbook.i_epoch)

def get_trackers(logbook,net):
    net_nodes = qg.analyse.list_nodes(net, verbose=False)
    rule2 = [qg.analyse.rule_linear_nodes]
    conv_nodes = qg.analyse.find_nodes(net_nodes, rule2, mix='and')
    conv_modules = [n.name for n in conv_nodes]
    # print(conv_modules)
    conv_trackers = [qm.WeightUpdateTracker(logbook.writer,conv_modules), qm.LinearOutputTracker(logbook.writer,conv_modules)]
    return conv_trackers


def froze_net(logbook,net):
    """freezes certain layers"""

    total_zeros, params = 0, 0

    for name, param in net.named_parameters():

        if name.find("head2") != -1 or name.find("head3") != -1 or name.find("head4") != -1:
            print(name)
        else:
            param.requires_grad = False

# def head_same(logbook,net):
#     """Logs Sparsity for each layer"""
#
#
#     for name, param in net.named_parameters():
#         if name.find("head2") != -1 or name.find("head3") != -1 or name.find("head4") != -1:
#             # param.requires_grad = False
#             pass
#         else:
#             model

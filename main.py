import argparse
import horovod.torch as hvd
import torch
import torch.quantization
from experiments import Logbook
from experiments import get_data, get_network, get_training,analyze_net,get_trackers, froze_net
from experiments import train, validate,train_6D,valid_6D
from problems.SwissCube.Widedepth.utils import visualize_accuracy_per_depth



# Command Line Interface
parser = argparse.ArgumentParser(description='QuantLab')
parser.add_argument('--problem',    help='Data set')
parser.add_argument('--topology',   help='Network topology')
parser.add_argument('--exp_id',     help='Experiment to launch/resume/test',                                            default=None)
parser.add_argument('--new_exp',    help='Whether it should continue the same experiment or create a new experiment',   default=True)
parser.add_argument('--mode',       help='Mode: train/test',                                                            default='train')
parser.add_argument('--ckpt_every', help='Frequency of checkpoints (in epochs)',                                        default=1)
parser.add_argument('--starting_ckpt',      help='specify from which epoch to continue, could be a specific number, "best" or "last" checkpoint',   default="last")
args = parser.parse_args()

# initialise Horovod
hvd.init()

# create/retrieve experiment logbook
logbook = Logbook(args.problem, args.topology, args.exp_id,args.new_exp)

# run experiment
if args.mode == 'train':

    logbook.get_training_status()

    for i_fold in range(logbook.i_fold, logbook.config['experiment']['n_folds']):

        # create data sets, network, and training algorithm
        train_l, valid_l = get_data(logbook)

        net = get_network(logbook)
        loss_fn, opt, lr_sched, ctrls = get_training(logbook, net)
        # boot logging instrumentation and load most recent checkpoint
        logbook.open_fold()

        logbook.i_epoch=0
        lr_sched.step(logbook.i_epoch)
        trackers= get_trackers(logbook,net)

        for i_epoch in range(logbook.i_epoch, logbook.config['experiment']['n_epochs']):


            if logbook.config["experiment"]["task"]=="classification":
                train_stats = train(logbook, train_l, net, loss_fn, opt, ctrls)
                valid_stats = validate(logbook, valid_l, net, loss_fn, ctrls)
                analyze_net(logbook, net)
                stats = {**train_stats, **valid_stats}
            elif logbook.config["experiment"]["task"]=="6D":

                loss_cls, loss_reg = train_6D(train_l, net, opt,logbook,ctrls,trackers)
                # analyze_net(logbook, net)
                logbook.writer.add_scalar('training/learning_rate', opt.param_groups[0]['lr'], logbook.i_epoch)
                logbook.writer.add_scalar('training/loss_cls', loss_cls, logbook.i_epoch)
                logbook.writer.add_scalar('training/loss_reg', loss_reg, logbook.i_epoch)
                logbook.writer.add_scalar('training/loss_all', (loss_cls + loss_reg), logbook.i_epoch)
                accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range = valid_6D( valid_l, net, logbook)

            else:
                train_stats = train(logbook, train_l, net, loss_fn, opt, ctrls)
                stats = {**train_stats}
            
            if 'metrics' in lr_sched.step.__code__.co_varnames:
                lr_sched_metric = stats[logbook.config['training']['lr_scheduler']['step_metric']]
                lr_sched.step(lr_sched_metric)
            else:
                lr_sched.step()


            visualize_accuracy_per_depth(
                accuracy_adi_per_class,
                accuracy_rep_per_class,
                accuracy_adi_per_depth,
                accuracy_rep_per_depth,
                depth_range)


            # save model if last/checkpoint epoch, and/or if update metric has improved
            is_last_epoch = (logbook.i_epoch + 1) == logbook.config['experiment']['n_epochs']
            is_ckpt_epoch = ((logbook.i_epoch + 1) % args.ckpt_every) == 0

            if is_last_epoch or is_ckpt_epoch:
                logbook.store_checkpoint(net, opt, lr_sched, ctrls)

            logbook.i_epoch += 1

        import numpy as np
        if logbook.is_master:
            print('Images per second (training, batch size {}): {} +- {}'.format(logbook.config['data']['bs_train'], np.mean(measurements['training']), 1.96 * np.std(measurements['training'])))
            print('Images per second (validation, batch size {}): {} +- {}'.format(logbook.config['data']['bs_valid'], np.mean(measurements['validation']), 1.96 * np.std(measurements['validation'])))

        logbook.close_fold()

if args.mode == 'test':
    logbook.get_training_status()
    # create data sets, network, and training algorithm
    train_l, valid_l = get_data(logbook)

    net = get_network(logbook)
    loss_fn, opt, lr_sched, ctrls = get_training(logbook, net)

    logbook.open_fold()

    logbook.i_epoch = 0
    print("loaded checkpoint")
    lr_sched.step(logbook.i_epoch)
    checkpoint = torch.load("PATH_TO_IMQ_6D_CKPT from checkpoints.zip")
    net.load_state_dict(checkpoint["network"])
    valid_6D(valid_l, net, logbook)

    logbook.close_fold()

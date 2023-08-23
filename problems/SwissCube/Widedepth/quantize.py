import quantlab.graphs as qg
import quantlab.algorithms as qa


__all__ = ['layers_ste_inq', 'layers_ste_inq_get_controllers']


def layers_ste_inq(config, net):


    def get_layers_conv_nodes(net):

        net_nodes = qg.analyse.list_nodes(net, verbose=False)

        nodes_backbone = qg.analyse.rule_single_block_except(net_nodes,'model.backbone')
        rule2 = [qg.analyse.rule_linear_nodes]
        conv_nodes_backbone = qg.analyse.find_nodes(nodes_backbone, rule2, mix='and')

        nodes_head = qg.analyse.rule_single_block_except(net_nodes,'model.head')
        rule2 = [qg.analyse.rule_linear_nodes]
        conv_nodes_head = qg.analyse.find_nodes(nodes_head, rule2, mix='and')

        nodes_fpn = qg.analyse.rule_single_block_except(net_nodes, 'fpn')
        conv_nodes_fpn = qg.analyse.find_nodes(nodes_fpn, rule2, mix='and')

        if config["first_layer"]==False:
            conv_nodes_backbone.pop(0)

        if config["last_layer"]==False:
            conv_nodes_head.pop()

        return conv_nodes_head,conv_nodes_backbone,conv_nodes_fpn

    inq_config_backbone = config['INQB']
    inq_config_fpn = config['INQF']
    inq_config_head = config['INQH']


    # replace convolutions with INQ convolutions
    conv_nodes_head,conv_nodes_backbone,conv_nodes_fpn = get_layers_conv_nodes(net)


    ### Uncomment below if you want to quantize backbone
    # qg.edit.replace_linear_inq_wide_depth(net, conv_nodes_backbone, num_levels=inq_config_backbone['n_levels'], quant_init_method=inq_config_backbone['quant_init_method'], quant_strategy=inq_config_backbone['quant_strategy'])

    ### Uncomment below if you want to quantize fpn
    qg.edit.replace_linear_inq_wide_depth(net, conv_nodes_fpn, num_levels=inq_config_fpn['n_levels'], quant_init_method=inq_config_fpn['quant_init_method'], quant_strategy=inq_config_fpn['quant_strategy'])

    ### Uncomment below if you want to quantize head
    # qg.edit.replace_linear_inq_wide_depth(net, conv_nodes_head, num_levels=inq_config_head['n_levels'], quant_init_method=inq_config_head['quant_init_method'], quant_strategy=inq_config_head['quant_strategy'])

    ###Uncomment all the lines above lines if you want to quantize the whole network.

    return net


def layers_ste_inq_get_controllers(config, net):

    net_nodes = qg.analyse.list_nodes(net)


    ##Used for activations
    # get STE controller
    ste_ctrl_config = config['STE']
    ste_modules = qa.ste.STEController.get_ste_modules(net_nodes)
    ste_controller = qa.ste.STEController(ste_modules, ste_ctrl_config['clear_optim_state_on_step'])

    # get INQ controller
    inq_ctrl_config = config['INQ']
    inq_modules = qa.inq.INQController.get_inq_modules(net_nodes)
    inq_controller = qa.inq.INQController(inq_modules, inq_ctrl_config['schedule'],
                                          inq_ctrl_config['clear_optim_state_on_step'])

    return [ste_controller, inq_controller]

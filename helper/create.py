'''
create the optimizer and scheduler used in train.py
'''
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


def create_optimizer(model, opt):
    if opt.optimizer_name == "adam":
        optimizer = Adam(model.parameters(),
                         lr=opt.lr,
                         eps=opt.opt_eps,
                         weight_decay=opt.weight_decay)
        if opt.betas:
            optimizer.betas = opt.betas
    elif opt.optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(),
                          lr=opt.lr,
                          eps=opt.opt_eps,
                          weight_decay=opt.weight_decay)
        if opt.betas:
            optimizer.betas = opt.betas
    elif opt.optimizer_name == "sgd":
        optimizer = SGD(model.parameters(),
                        lr=opt.lr,
                        weight_decay=opt.weight_decay,
                        momentum=opt.momentum,
                        nesterov=True)
    elif opt.optimizer_name == "rmsprop":
        optimizer = RMSprop(model.parameters(),
                            lr=opt.lr,
                            eps=opt.opt_eps,
                            weight_decay=opt.weight_decay,
                            momentum=opt.momentum)
    else:
        raise NotImplementedError('optimizer type {} does not support'.format(opt.optimizer_name))
    return optimizer


def create_scheduler(optimizer, opt):
    if opt.lr_scheduler == "step":
        lr_scheduler = StepLR(optimizer,
                              step_size=50,
                              gamma=opt.lr_decay_rate)
    elif opt.lr_scheduler == "mstep":
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=opt.lr_decay_epochs,
                                   gamma=opt.lr_decay_rate)
    elif opt.lr_scheduler == "exp":
        lr_scheduler = ExponentialLR(optimizer,
                                     gamma=opt.lr_decay_rate)
    elif opt.lr_scheduler == "cos":
        lr_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=50,
                                         eta_min=1e-7)
    elif opt.lr_scheduler == "reduce":
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         mode='min',
                                         factor=opt.lr_decay_rate,
                                         patience=10,
                                         verbose=False,
                                         threshold=0.0001,
                                         threshold_mode="rel",
                                         eps=opt.opt_eps)
    else:
        raise NotImplementedError('lr_scheduler type {} does not support'.format(opt.lr_scheduler))
    return lr_scheduler
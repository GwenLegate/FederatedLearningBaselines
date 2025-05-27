from torch.optim import SGD
from torch.optim import Adam
def get_optimizer(args, model):
    if args.fed_type.lower() == 'fedavgm':
        return SGD(model.parameters(), args.server_lr, args.momentum)
    elif args.fed_type.lower() == 'fedadam':
        return Adam(model.parameters(), lr=args.server_lr, betas=(args.beta1, args.beta2), weight_decay=1e-5, eps=args.adam_eps)
    else:
        return SGD(model.parameters(), args.server_lr)
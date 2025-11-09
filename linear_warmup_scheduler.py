from torch.optim.lr_scheduler import LambdaLR


def linear_warmup_scheduler(opt, steps):
    def lr_lambda(current_step):
        return min(current_step/max(steps, 1), 1)
    return LambdaLR(opt, lr_lambda)
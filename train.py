import fire
import wandb
import math
import torch
import torch.nn.functional as F
from data import CifarLoader
from cifarnet import CifarNet
from resnet import ResNet
from tqdm.auto import tqdm


# learning rate schedule: linear warmup + cosine decay
def warmup_cosine_lr_schedule(peak_lr, num_steps, warmup_frac=0.1):
    cooldown_frac = 1 - warmup_frac
    def schedule(step: int):
        r = step / num_steps # relative progress in training
        assert 0 <= r <= 1
        if r < warmup_frac:
            return (r / warmup_frac) * peak_lr
        else:
            cosine_scaling = (1 + math.cos(math.pi * (r-warmup_frac)/cooldown_frac))/2 # [0, 1]
            return cosine_scaling * peak_lr
    return schedule


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    test_images = loader.normalize(loader.images)
    logits = torch.cat([model(inputs).clone() for inputs in test_images.split(2000)])
    return (logits.argmax(1) == loader.labels).float().mean().item()


def train(
    model_name = 'resnet20',
    peak_lr = 0.01,
    momentum = 0.9,
    batch_size = 1024,
    n_epochs = 100,
    corrupt_frac = 0,
    wandb_mode = 'offline',
    run_name = None,
    **kwargs,
):
    if len(kwargs) > 0: raise NameError(f'Unrecognized arguments: {kwargs}')
    train_config = locals()

    # dataset
    test_loader = CifarLoader("/tmp/cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("/tmp/cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2), corrupt_frac=corrupt_frac)

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.device(device):
        model = ResNet(model_name) if model_name.startswith('resnet') else CifarNet()
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:_}')

    # initialize whitening layer using training images
    if model_name == 'cifarnet':
        whiten_bias_train_steps = math.ceil(3 * len(train_loader))
        model.reset()
        train_images = train_loader.normalize(train_loader.images[:5000])
        model.init_whiten(train_images)

    # optimizer
    n_steps = n_epochs * len(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, nesterov=True)
    lr_schedule = warmup_cosine_lr_schedule(peak_lr, n_steps)

    # wandb
    wandb.init(project='noise-1', config=train_config, mode=wandb_mode, name=run_name)
    wandb.summary.update(dict(n_params=n_params))

    step = 0
    for epoch in (pbar := tqdm(range(n_epochs))):

        # train for 1 epoch
        model.train()
        for inputs, labels in train_loader:

            # update learning rate
            for group in optimizer.param_groups:
                group['lr'] = lr_schedule(step)

            # training step
            # outputs = model(inputs, whiten_bias_grad=False)
            outputs = model(inputs)
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            step += 1

        # eval
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader)
        pbar.set_postfix_str(f'{train_acc=:.1%}, {val_acc=:.1%}')
        wandb.log(dict(train_acc=train_acc, val_acc=val_acc), epoch)
    wandb.finish()


if __name__ == '__main__':
    fire.Fire(train)

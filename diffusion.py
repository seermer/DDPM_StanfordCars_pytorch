import sys
import tqdm
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms as T

import models
from configs import *
from data import CarDataset
from helper import seed_all, save_imgs


class Diffusion:
    def __init__(self, model, optim=None):
        self.model = model.to(DEVICE)
        self.loss_fn = nn.MSELoss()
        self.optim = optim
        self.scaler = torch.cuda.amp.GradScaler(enabled=AMP)

        if CHANNELS_LAST:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model = torch.compile(self.model)

        betas = torch.linspace(*NOISE, MAX_STEPS, device=DEVICE)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.img_coefs = torch.sqrt(alphas_bar)
        self.noise_coefs = torch.sqrt(1. - alphas_bar)

        with torch.inference_mode():
            self.sqrt_alpha_recip = 1. / torch.sqrt(alphas)
            self.pred_coefs = (1. - alphas) / self.noise_coefs
            self.post_std = torch.sqrt(betas)

    @torch.compile()
    def add_noise(self, imgs, times):
        img_scales = self.img_coefs[times, None, None, None]
        noise_scales = self.noise_coefs[times, None, None, None]

        noises = torch.randn_like(imgs)
        new_imgs = imgs * img_scales + noises * noise_scales
        return new_imgs, noises

    @torch.compile()
    def sample(self, dataset,
               num_samples=NUM_SAMPLES, sample_space=SAMPLE_SPACE):
        self.model.eval()
        x = torch.randn((num_samples, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
        idx = torch.ones((num_samples,), dtype=torch.int64, device=DEVICE)
        conds = dataset.get_rand_cond((num_samples,)).to(DEVICE)
        retvals = []

        if CHANNELS_LAST:
            x = x.to(memory_format=torch.channels_last)

        for step in reversed(range(0, MAX_STEPS)):
            t = idx * step
            z = torch.randn_like(x) if step > 0 else 0.
            x = self.sqrt_alpha_recip[t, None, None, None] * (
                    x -
                    self.pred_coefs[t, None, None, None] *
                    self.model(x, t, conds)
            ) + self.post_std[t, None, None, None] * z
            # NOTE: some people also use beta tilde as posterior variance,
            # which is empirically similar according to the original paper
            if step % sample_space == 0 or step == 1 or step == MAX_STEPS - 1:
                retvals.append(x / 2. + 0.5)

        return retvals

    def train(self, loader, epochs):
        ema = models.EMA(EMA_SCALE)

        losses = []
        lowest_loss = float('inf')
        for epoch in range(1, epochs + 1):
            print('epoch', epoch, flush=True)
            self.model.train()
            total_loss = 0.
            for img, conds in tqdm.tqdm(loader, total=len(loader), file=sys.stdout):
                img = img.to(DEVICE, non_blocking=True)
                conds = conds.to(DEVICE, non_blocking=True).to(torch.int64)
                if CHANNELS_LAST:
                    img = img.to(memory_format=torch.channels_last)
                times = torch.randint(1, MAX_STEPS, (img.size(0),),
                                      device=DEVICE)
                with torch.no_grad():
                    img *= 2.
                    img -= 1.
                    img, noise = self.add_noise(img, times)

                self.optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=AMP):
                    pred = self.model(img, times, conds)
                    loss = self.loss_fn(pred, noise)

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

                self.scaler.step(self.optim)
                self.scaler.update()

                ema.update(self.model)

                with torch.inference_mode():
                    total_loss += loss.detach() * img.size(0)

            with torch.inference_mode():
                avg_loss = (total_loss / len(loader.dataset)).cpu().numpy()
                losses.append(avg_loss)
                print(f'avg loss {avg_loss}', flush=True)
                if epoch % 15 == 1 or epoch == epochs - 1:
                    t = time.time()
                    with torch.cuda.amp.autocast(enabled=AMP):
                        sampled = self.sample(loader.dataset,
                                              num_samples=NUM_SAMPLES, sample_space=SAMPLE_SPACE)
                    print(f'sampling done in {time.time() - t} seconds')
                    save_imgs(sampled, f'epoch{epoch}.png', './samples')
                    torch.save(self.model.state_dict(), './ckpts/model.pth')
                elif avg_loss <= lowest_loss:
                    print('new best model', flush=True)
                    torch.save(self.model.state_dict(), './ckpts/model.pth')
                    lowest_loss = avg_loss
        return losses


def main():
    transforms_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    ds = CarDataset(transform=transforms_train,
                    root='./cars', download=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=6, pin_memory=True, persistent_workers=True)
    model = models.DiffusionModel(max_step=MAX_STEPS,
                                  init_channels=INIT_CHANNELS,
                                  num_res=NUM_RES,
                                  dropout=DROPOUT,
                                  cond_len=1,
                                  cond_num_emb=len(ds.classes))
    print('initialized model with size',
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    diffuser = Diffusion(model, optim)
    losses = diffuser.train(loader, 800)
    plt.plot(list(range(1, 800 + 1)), losses)
    plt.savefig('loss.png')


if __name__ == '__main__':
    seed_all(SEED)
    main()

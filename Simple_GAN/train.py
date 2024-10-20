import torch
from config import Config
from archi import Drisciminator, Generator
import torchvision

config = Config()
step = 0

print("utilizing  ------------- ----- >>>" + config.device)

for epoch in range(config.epochs):
    for batch_idx, (real, _) in enumerate(config.loader):
        real = real.view(-1, 784).to(config.device)
        batch_size = real.shape[0]

        ### TRAINING DISCRIMINATOR: maximize log(D(real)) + log(1- D(G(z)))
        noise = torch.randn(config.batch_size, config.z_dim).to(config.device)
        fake = config.gen(noise)
        disc_real = config.disc(real).view(-1)
        lossD_real = config.criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = config.disc(fake).view(-1)
        lossD_fake = config.criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        config.disc.zero_grad()
        lossD.backward(retain_graph = True)
        config.optim_disc.step()

        ### Train Generato: minimize log(1- D(G(z))) <-or-> maxmize log(D(G(z)))
        output = config.disc(fake).view(-1)
        lossG = config.criterion(output, torch.ones_like(output))
        config.gen.zero_grad()
        lossG.backward()
        config.optim_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{config.epochs}] Batch {batch_idx}/{len(config.loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = config.gen(config.fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                config.writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                config.writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

        
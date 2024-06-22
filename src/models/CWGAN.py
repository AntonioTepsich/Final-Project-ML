import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cwgan.generator import Generator
from src.models.cwgan.discriminator import Critic


class CWGAN(nn.Module):
    # def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10):
    def __init__(self, params):
        super().__init__()
        
        
        self.display_step = params.display_step
        self.lambda_recon = params.lambda_recon
        self.lambda_gp = params.lambda_gp
        self.lambda_r1 = params.lambda_r1
        self.device = params.device
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.learning_rate = params.learning_rate
        
        self.generator = Generator(self.in_channels, self.out_channels).to(self.device)
        self.critic = Critic(self.in_channels + self.out_channels).to(self.device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = optim.Adam(self.critic.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        self.recon_criterion = nn.L1Loss()
        self.generator_losses, self.critic_losses  =[],[]

        self.current_epoch = 0



        self.step_count = 0
    
    # def configure_optimizers(self):
    #     return [self.optimizer_C, self.optimizer_G]
        
    def generator_step(self, real_images, conditioned_images):
        # WGAN has only a reconstruction loss
        self.optimizer_G.zero_grad()
        fake_images = self.generator(conditioned_images)
        recon_loss = self.recon_criterion(fake_images, real_images)
        recon_loss.backward()
        self.optimizer_G.step()
        
        # Keep track of the average generator loss
        self.generator_losses += [recon_loss.item()]
        
        
    def critic_step(self, real_images, conditioned_images):
        self.optimizer_C.zero_grad()  # Reinicia los gradientes del optimizador del crítico.
        fake_images = self.generator(conditioned_images).detach()  # Detiene la propagación de gradientes.
        fake_logits = self.critic(fake_images, conditioned_images)  # Evalúa las imágenes falsas.
        real_logits = self.critic(real_images, conditioned_images)  # Evalúa las imágenes reales.

        # Calcula la pérdida básica del crítico.
        loss_C = real_logits.mean() - fake_logits.mean()

        # Penalización por gradiente (Gradient penalty)
        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=self.device, requires_grad=True)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        interpolated_logits = self.critic(interpolated, conditioned_images)
        grad_outputs = torch.ones(interpolated_logits.size(), device=self.device, requires_grad=False)
        gradients = torch.autograd.grad(outputs=interpolated_logits,
                                        inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradient_penalty  # Agrega la penalización por gradiente a la pérdida del crítico.


        # Backpropagation
        loss_C.backward()
        self.optimizer_C.step()
        self.critic_losses += [loss_C.item()]
        
    def train_step(self, batch):
        condition, real = batch
        condition = condition.to(self.device)
        real = real.to(self.device)
        
        if self.step_count%2 == 0:
            self.critic_step(real, condition)
        else:
            self.generator_step(real, condition)
        self.step_count += 1

        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        
        # mover esto al training loop
        # if self.current_epoch%self.display_step==0 and self.step_count%2==1:
            # fake = self.generator(condition).detach()
            # torch.save(self.generator.state_dict(), "ResUnet_"+ str(self.current_epoch) +".pt")
            # torch.save(self.critic.state_dict(), "PatchGAN_"+ str(self.current_epoch) +".pt")
            # display_progress(condition[0], real[0], fake[0], self.current_epoch)
        
        print(f"Epoch {self.current_epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
        self.current_epoch += 1
        return gen_mean
    
    def eval_step(self, batch):
        condition, real = batch
        real = real.to(self.device)
        condition = condition.to(self.device)
        fake = self.generator(condition).detach().squeeze().permute(1, 2, 0)
        fake = fake.to(self.device)
        condition = condition.detach().squeeze(0).permute(1, 2, 0)
        real = real.detach().squeeze(0).permute(1, 2, 0)
        recon_loss = self.recon_criterion(fake, real)
        return recon_loss.item()

    def predict(self, condition):
        condition = condition.to(self.device)
        return self.generator(condition)




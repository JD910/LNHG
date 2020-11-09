import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


def Train_Disc(self, free_img, noised_img, train=True):


    self.D_optimizer.zero_grad() 
    z = Variable(noised_img)
    #real_img = Variable(free_img / 4096)  
    real_img = Variable(free_img)  
    if self.gpu:
        z = z.cuda()
        real_img = real_img.cuda()
    #fake_img = self.generator(z,real_img,Disc=True)
    fake_img = self.generator(z)

    real_validity = self.discriminator(real_img)
    #fake_validity = self.discriminator(fake_img.data / 4096) 
    fake_validity = self.discriminator(fake_img.data) 
    gradient_penalty = calc_gradient_penalty(self, real_img.data, fake_img.data)

    d_loss = self.lambda_d_real * torch.mean(-real_validity) + self.lambda_d_fake * torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
    if train:
        d_loss.backward()
        
        self.D_optimizer.step()

    return d_loss.data.item(), self.lambda_d_real * torch.mean(-real_validity).cpu().item(), self.lambda_d_fake * torch.mean(fake_validity).cpu().item(), self.lambda_gp * gradient_penalty.cpu().item()

def calc_gradient_penalty(self, free_img, gen_img):
    batch_size = free_img.size()[0]
    alpha = Variable(torch.rand(batch_size, 1))
    alpha = alpha.expand(batch_size, free_img.nelement(
    ) // batch_size).contiguous().view(free_img.size())
    if self.gpu:
        alpha = alpha.cuda()

    interpolates = (alpha * free_img + (1 - alpha)
                    * gen_img).requires_grad_(True)
    disc_interpolates = self.discriminator(interpolates)
    #fake = Variable(torch.Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = torch.ones(disc_interpolates.size())
    if self.gpu:
        fake = fake.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]



    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
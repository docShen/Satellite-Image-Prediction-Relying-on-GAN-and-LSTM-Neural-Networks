import argparse
import glob
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torch
from model import Generator,Discriminator
from Config import dic_obj as opt
os.makedirs("images", exist_ok=True)
print(opt)


class Dataset_m(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data_list = glob.glob(f'{self.root}/{opt.year}/*/*jpg')
        pass
    def __getitem__(self, index):
        data = self.data_list[index]
        data = Image.open(data).convert('L')
        data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':

    img_shape = (1, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    generator = Generator(input_dim=opt.latent_dim).cuda()
    discriminator = Discriminator().cuda()

    transforms_train = transforms.Compose([transforms.Resize((opt.img_size,opt.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=0.5, std=0.5),
                                           ])

    my_dataset = Dataset_m(f'{opt.data_path}',transform = transforms_train)


    dataloader = DataLoader(
        dataset = my_dataset,
        batch_size=opt.batch_size_GAN,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr_GAN)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr_GAN)
    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epochs_GAN):

        for i, imgs in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.cuda()
            iteration = epoch * len(dataloader) + i
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).cuda().float()

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            # if torch.rand(1) > 0.8:
            #     fake_of_D = torch.mean(discriminator(fake_imgs)) - (torch.abs(torch.rand(1))/5).cuda()
            #     true_of_D = torch.mean(discriminator(real_imgs)) + (torch.abs(torch.rand(1))/5).cuda()
            # else:
            fake_of_D = torch.mean(discriminator(fake_imgs))
            true_of_D = torch.mean(discriminator(real_imgs))

            loss_D =  fake_of_D - true_of_D #改进2、生成器和判别器的loss不取log

            loss_D.backward()
            optimizer_D.step() #只更新discriminator的参数

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss

                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()#只更新 generator 的参数

                print(
                    "[Epoch %d/%d] [iteration %d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.epochs_GAN, iteration, loss_D.item(), loss_G.item())
                )

        if (epoch+1 ) % 200 == 0 :
            save_model_path = './saved_model'
            os.makedirs(save_model_path,exist_ok=True)
            torch.save(generator.state_dict(), save_model_path + f'/my_G{epoch+1}.pth')
            save_image(gen_imgs.data[:16], f"images/{epoch+1}.png" , nrow=4, normalize=True)

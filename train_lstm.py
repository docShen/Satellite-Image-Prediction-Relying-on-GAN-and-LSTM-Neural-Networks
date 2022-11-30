import random
import torch
import glob
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torch
from model import lstm ,Generator
from Config import dic_obj as opt

class Dataset_m(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data_file_list = sorted(glob.glob(f'{self.root}/{opt.year}/*'))
        pass
    def __getitem__(self, index):
        cur_len = 0
        while cur_len <= opt.J:
            self.sel_data_list = sorted(glob.glob(f'{self.data_file_list[index]}/*jpg'))
            cur_len = len(self.sel_data_list)
            index += 1

        data_idx = int(np.random.choice(range(0,cur_len),1))

        while data_idx+opt.J >= len(self.sel_data_list):
            data_idx = data_idx - 1

        sel_data_list = []
        for i in range(opt.J + 1):
            data = Image.open(self.sel_data_list[data_idx + i]).convert('L')
            data = self.transform(data).flatten(1,-1)
            sel_data_list.append(data)
        data_mat = torch.cat(sel_data_list,dim=0)

        assert len(sel_data_list) == opt.J + 1 , 'error ！'

        return data_mat[:opt.J,:], data_mat[-1,:]

    def __len__(self):
        return len(self.data_file_list)


if __name__ == '__main__':

    transforms_train = transforms.Compose([transforms.Resize((opt.img_size,opt.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=0.5, std=0.5),
                                           ])

    my_dataset = Dataset_m('./my_data/download/W. South Pacific',transform = transforms_train)


    dataloader = DataLoader(
        dataset = my_dataset,
        batch_size=opt.batch_size_lstm,
        shuffle=True,
        drop_last=True,
    )

    LSTM = lstm(input_size=opt.img_size * opt.img_size,output_size=opt.latent_dim,hidden_size=opt.hidden_size,num_layer=opt.layer).cuda()
    optimizer = torch.optim.Adam(LSTM.parameters(), lr=opt.lr_lstm)

    G = Generator(input_dim=opt.latent_dim).cuda()
    G.load_state_dict(torch.load(f'./saved_model/{opt.load_G_name}'))
    for param in G.parameters():
        param.requires_grad = False

    critiron = torch.nn.MSELoss(reduction='mean')
    iteration = 0
    for epoch in range(opt.epochs_lstm):
        for i, (imgs,target) in enumerate(dataloader):
            optimizer.zero_grad()

            imgs = imgs.cuda()
            target = target.reshape(opt.batch_size_lstm,opt.img_size,opt.img_size).cuda()

            target_emb = LSTM(imgs)
            target_emb = target_emb[:,-1,:]

            G.eval()

            gen_img = G(target_emb)

            loss = critiron(gen_img.squeeze(1),target)

            loss.backward()
            optimizer.step()  # 只更新 LSTM 的参数
            if iteration % 100 == 0:
                print(
                    "[Epoch %d/%d] [iter %d] [loss: %f]"
                    % (epoch, opt.epochs_lstm, iteration, loss.item())
                )
            iteration += 1

            if (iteration+1 ) % 2000 == 0 :
                os.makedirs('./images_lstm', exist_ok=True)

                for i in range(opt.J):
                    save_image(imgs.data[0][i:i + 1].reshape(opt.img_size, opt.img_size),
                               f"images_lstm/E{iteration+1}_aseries_{i}.png", nrow=4, normalize=True)

                save_image(gen_img.data[0], f"images_lstm/E{iteration+1}_pred.png", nrow=4, normalize=True)

                save_model_path = './saved_model_LSTM'
                os.makedirs(save_model_path,exist_ok=True)
                torch.save(LSTM.state_dict(), save_model_path + f'/my_Lstm_E{iteration+1}.pth')
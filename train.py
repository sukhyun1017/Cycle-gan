import itertools
import argparse
from glob import glob
from tqdm.auto import tqdm
import wandb

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import weights_init_normal
from dataset import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--pretrained', type=bool, default=True, help='using pretrained model')
opt = parser.parse_args("")
print(opt)

#wandb
wandb.init(project='cycle_gan',config = {'archetecture':"cyclegan"})
config = wandb.config
#cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transforms
transforms_ = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
])

# dataset
trainA_list = glob('montage/*.png')
trainB_list = glob('L/*.*')
train_dataset = ImageDataset(trainA_list,trainB_list, transforms_=transforms_)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu)

#model
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

#use_pretrained_model
if opt.pretrained == True: 
    netG_A2B.load_state_dict(torch.load('netG_A2B_3.pt'))
    netG_B2A.load_state_dict(torch.load('netG_B2A_3.pt'))
    netD_A.load_state_dict(torch.load('netD_A_3.pt'))
    netD_B.load_state_dict(torch.load('netD_B_3.pt'))

# 가중치(weights) 초기화
'''
else:
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
'''

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)

#buffer
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# 손실 함수(loss function)
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Image
image= Image.open('sukhyun.png')
cropimage = image.crop((10,10,380,340))
myimage = transforms_(cropimage.convert('RGB')).to(device)

NUM_ACCUMULATION_STEPS = 8

for epoch in range(opt.n_epochs):
    for i, batch in enumerate(tqdm(train_dataloader)):
        # Set model input
        real_A = batch["A"].to(device)   
        real_B = batch["B"].to(device)  
        
        target_real = torch.cuda.FloatTensor(real_A.size(0), 1).fill_(1.0).to(device) # 진짜(real): 1
        target_fake = torch.cuda.FloatTensor(real_A.size(0), 1).fill_(0.0).to(device)# 가짜(fake): 0
        
        ###### Generators A2B and B2A ######
        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
            optimizer_G.zero_grad()
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
        	optimizer_G.step()

        ###################################

        ###### Discriminator A ######

        # Real loss
        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
        	optimizer_D_A.zero_grad()
         
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
        	optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######

        # Real loss
        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
        	optimizer_D_B.zero_grad()
         
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
        	optimizer_D_B.step()
            
        ###################################
        

    # 하나의 epoch이 끝날 때마다 저장
    finalimage = netG_A2B(myimage.detach())
    wandb.Image(to_pil_image(0.5*finalimage+0.5))
    
    wandb.log({"Loss_D_B": round(loss_D_B.item(),6), 
               "Loss_D_A": round(loss_D_A.item(),6),
               "Loss_G": round(loss_G.item(),6),
               "IMAGE": wandb.Image(to_pil_image(0.5*finalimage.detach().cpu()+0.5))})


torch.save(netG_A2B.state_dict(), 'netG_A2B_4.pt')
torch.save(netG_B2A.state_dict(), 'netG_B2A_4.pt')
torch.save(netD_A.state_dict(), 'netD_A_4.pt')
torch.save(netD_B.state_dict(), 'netD_B_4.pt')
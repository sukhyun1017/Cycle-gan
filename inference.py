from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import torch
from models import Generator

image= Image.open('sukhyun.png')
cropimage = image.crop((10,10,380,340))

transforms_ = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # 각채널별로 mean.std값 부여
])

myimage = transforms_(cropimage.convert('RGB'))

model = Generator()
model.load_state_dict(torch.load('netG_A2B_3.pt'))

final = model(myimage)

plt.imshow(to_pil_image(0.5*final+0.5)) #denormalize
plt.show()

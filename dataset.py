from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, dataA, dataB, transforms_):
        self.transform = transforms_
        
        self.files_A = dataA
        self.files_B = dataB

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]).convert('RGB'))
       
        item_B = self.transform(Image.open(self.files_B[index]).convert('RGB'))

        return {'A':item_A , 'B':item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B)) 

class Testing():
    def plus(self,a,b):
        
        return a+b
        
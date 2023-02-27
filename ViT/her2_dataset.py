import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class HER2Dataset(datasets.ImageFolder):
    def __init__(self, path : str, transform=None):
        super(HER2Dataset, self).__init__(path, transform=transform)
        self.path = path
        #self = datasets.ImageFolder(self.path, transform=transform)
        self.splitGenerated = False
    

    def genSplits(self, test_split_size : int):
        test_size = int(test_split_size * len(self))
        train_size = len(self) - test_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self, [train_size, test_size])
        self.splitGenerated = True
        return self.train_dataset, self.test_dataset


    def getDataLoaders(self, batch_size : int):
        if not self.splitGenerated:
            print("Splits not yet generated. Call genSplits() before this.")
            return None

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def getSampleCountByClass(self):
        train_classes = [self.targets[i] for i in self.train_dataset.indices]
        train_count = (list(map(lambda x: (x[0], x[1], self.classes[x[0]]), Counter(train_classes).items() )))
        test_classes = [self.targets[i] for i in self.test_dataset.indices]
        test_count = (list(map(lambda x: (x[0], x[1], self.classes[x[0]]), Counter(test_classes).items() )))

        return train_count, test_count


if __name__=="__main__":
    from torchvision.transforms import ToTensor
    
    DATASET_PATH='../datasets/HER2_gastric_5classes'
    transform = ToTensor()

    dataset = HER2Dataset(DATASET_PATH, transform)
    train_dataset, test_dataset = dataset.genSplits(0.2)
    train_loader, test_loader = dataset.getDataLoaders(2)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    print(images.shape)
    
    from PIL import Image

    with Image.open("dog.jpg") as im:
        print(transform(im).shape)



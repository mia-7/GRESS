import torchvision

class Transforms:
    def __init__(self):

        self.train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomAffine(degrees=7,
                                    translate=(0, 0.07),  # horizontal and vertical shifts
                                    shear=7,
                                    scale=(1, 1.2)  # zoom range
                                    )])

    def __call__(self, x):
        data1 = x
        data2 = self.train_transform(x)
        return data1, data2


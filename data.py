from torchvision import datasets


class CarDataset(datasets.StanfordCars):
    def get_rand_cond(self, size):
        import torch
        return torch.randint(len(self.classes), size=size, dtype=torch.int64)


class CIFAR10(datasets.CIFAR10):
    def get_rand_cond(self, size):
        import torch
        return torch.randint(len(self.classes), size=size, dtype=torch.int64)

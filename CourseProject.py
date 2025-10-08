from torchvision.datasets import CelebA
train_dataset = CelebA(root='./data4/', split='train', download=True)
test_dataset = CelebA(root='./data4/', split='test', download=True)

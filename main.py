import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gpnn import GPNN
from om import om
from acr import ACR

class CapsuleNetwork(pl.LightningModule):
    def __init__(self,
                 image_size=28,
                 template_size=11,
                 num_template=9):
        super(CapsuleNetwork, self).__init__()
        self.num_acrs = num_template
        self.batch_size = 64
        self.noise = 4.
        self.gpnn = GPNN(n_in=28*28, n_hidden=50, n_out=self.num_acrs*7)
        self.acr = ACR(num_template, template_size, image_size)

    def forward(self, image):
        return self.gpnn(image)

    def training_step(self, batch, batch_idx):
        image, label = batch
        image_flatten = image.view(image.size(0), -1)
        out = self(image_flatten)
        out = out.view(out.size(0), self.num_acrs, 7)
        pose, intensity = out.split((6, 1), dim=-1)
        intensity = torch.sigmoid(intensity + (torch.rand_like(intensity) * self.noise - 2))
        res = self.acr(pose, intensity)
        render = om(res.transformed_templates)
        rec_loss = (0.5 * (render - image) ** 2).view(image.size(0), -1).sum(dim=-1).mean()

        # each image is expression by 4 templates
        l1_loss = 0.
        l1_loss += (intensity.abs().sum(-1).sum(-1) - 7).pow(2).mean()

        # each template is used balanced
        l1_loss1 = 0.
        l1_loss1 += (intensity.abs().sum(dim=0) - intensity.size(0)*7/9).pow(2).mean()

        loss = rec_loss + 100 * l1_loss + l1_loss1
        self.log('loss', loss)
        self.log('rec_loss', rec_loss)
        self.log('l1_loss', l1_loss)
        self.log('l1_loss1', l1_loss1)
        if batch_idx == 0:
            recon = [image[:8].cpu().detach(), render[:8].cpu().detach()]
            recon = torch.cat(recon, dim=0)
            rg = torchvision.utils.make_grid(
                recon,
                nrow=8, pad_value=0, padding=1
            )
            self.logger.experiment.add_image('recons', rg, self.current_epoch)

            temp = res.templates[0].unsqueeze(dim=1)
            rg = torchvision.utils.make_grid(temp, nrow=self.num_acrs, padding=1, pad_value=0)
            self.logger.experiment.add_image('templates', rg, self.current_epoch)

            trans_temp = res.transformed_templates.cpu().detach()[:8].view(8*9, 1, 28, 28)
            rg = torchvision.utils.make_grid(trans_temp, nrow=self.num_acrs, padding=1, pad_value=0)
            self.logger.experiment.add_image('trans_templates', rg, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        image_flatten = image.view(image.size(0), -1)
        out = self(image_flatten)
        out = out.view(out.size(0), self.num_acrs, 7)
        pose, intensity = out.split((6, 1), dim=-1)
        intensity = torch.sigmoid(intensity)
        res = self.acr(pose, intensity)
        render = om(res.transformed_templates)
        loss = (0.5 * (render - image) ** 2).view(image.size(0), -1).sum(dim=-1).mean()

        self.log('val_loss', loss)
        if batch_idx == 0:
            n = min(24, self.batch_size)
            recon = [image[:n].cpu().detach(), render[:n].cpu().detach()]
            recon = torch.cat(recon, dim=0)
            rg = torchvision.utils.make_grid(
                recon,
                nrow=8, pad_value=0, padding=1
            )
            self.logger.experiment.add_image('val_recons', rg, self.current_epoch)

            temp = res.templates[0].unsqueeze(dim=1)
            rg = torchvision.utils.make_grid(temp, nrow=self.num_acrs, padding=1, pad_value=0)
            self.logger.experiment.add_image('val_templates', rg, self.current_epoch)

            trans_temp = res.transformed_templates.cpu().detach()[:8].view(8*9, 1, 28, 28)
            rg = torchvision.utils.make_grid(trans_temp, nrow=self.num_acrs, padding=1, pad_value=0)
            self.logger.experiment.add_image('val_trans_templates', rg, self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(),
                                        lr=0.0001,
                                        momentum=0.9,
                                        weight_decay=0.0001)
        return [optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=1)

    def make_transforms(self):
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        return transforms

    def prepare_data(self):
        # train and validation datasets
        mnist_train = MNIST('./data', train=True, download=True, transform=self.make_transforms())
        # mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # test dataset
        mnist_test = MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

        # assign to use in data loaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_test
        self.test_dataset = mnist_test

if __name__ == "__main__":
    model = CapsuleNetwork()
    trainer = pl.Trainer(max_epochs=1000,
                         gpus=[0],
                         gradient_clip_val=2.)
    trainer.fit(model)

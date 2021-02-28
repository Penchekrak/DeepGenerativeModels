import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm.notebook import tqdm, trange
import wandb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def run(model, n_epochs, train_loader, test_loader, name, 
        mode='ae', sparse_ae=False):
    device = next(model.parameters()).device
    run = wandb.init(name=name, 
                     project='gen_models_hw1', 
                     tags=[mode, str(device)])
    wandb.watch(model)
    train(model, n_epochs, train_loader, name, mode, sparse_ae)
    if mode == 'ae':
        plt.figure(figsize=(15, 10))
        plt.axis('off')
        wandb.log({'Decoded': 
                   plt.imshow(decode_imgs(model, test_loader).cpu())})
        plt.figure(figsize=(15, 10))
        plt.axis('off')
        wandb.log({'Generated':
                   plt.imshow(generate_imgs(model, test_loader).cpu())})
    else:
        acc = check_accuracy(model, test_loader)
        wandb.log({"Test accuracy": acc})
    run.finish()


def train(model, n_epochs, train_loader, name, mode='ae', sparse_ae=False):
    if mode != 'ae' and sparse_ae:
        raise ValueError('Sparse mode only for ae')
    device = next(model.parameters()).device
    model.train()
    if mode == 'ae':
        loss_func = nn.BCELoss()
    else:
        loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in trange(n_epochs, desc='train loop', leave=True):
        for image, label in tqdm(train_loader, desc=f'epoch {epoch}' , leave=False):
            image = image.to(device)
            label = label.to(device)
            if mode == 'ae':
                output, l1_loss = model(image, sparse_ae)
                loss = loss_func(output, image) + 0.001 * l1_loss
            else:
                output = model(image)
                loss = loss_func(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            wandb.log({
                "Epoch": epoch, 
                "Train Loss": loss})

        path = 'params/' + name + str(epoch) + '.pt'
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, path)


@torch.no_grad()
def decode_imgs(ae, test_loader, max_images=10):
    device = next(ae.parameters()).device
    test_batch = next(iter(test_loader))[0][:max_images].to(device)
    ae.eval()
    decoded, _ = ae(test_batch)
    concated = torch.cat([test_batch, decoded])
    concated = make_grid(concated, test_batch.shape[0])[0]
    return concated


@torch.no_grad()
def generate_imgs(ae, test_loader, max_images=10):
    ae.eval()
    device = next(ae.parameters()).device
    test_batch = next(iter(test_loader))[0][:max_images].to(device)
    images_latent = ae.get_latent_features(test_batch)
    test_noise = (torch.randn_like(images_latent) * 0.9).to(device)
    sampled = torch.sigmoid(ae.decoder(test_noise.to(device))).to('cpu')
    return make_grid(sampled, test_batch.shape[0])[0]


@torch.no_grad()
def check_accuracy(clf, test_loader):
    device = next(clf.parameters()).device
    true_labels = []
    clf_predict_labels = []
    clf.eval() 

    for image, label in tqdm(test_loader, desc='test clf loop', leave=True):
    # YOUR CODE
        image = image.to(device)
        label = label.to(device)
        output = clf(image)
        true_labels.append(label.cpu())
        clf_predict_labels.append(torch.max(output.data, 1)[1].cpu())

    true_labels = torch.cat(true_labels, dim=0).numpy()
    clf_predict_labels = torch.cat(clf_predict_labels, dim=0).numpy()
    return accuracy_score(clf_predict_labels, true_labels)

@torch.no_grad()
def get_latent_data(ae, loader, train=True):
    ae.eval()
    device = next(ae.parameters()).device
    latent = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        lat = ae.get_latent_features(images).cpu().view(images.shape[0], -1)
        latent.append(lat)
        all_labels.append(labels)
    latent = torch.cat(latent, 0)
    all_labels = torch.cat(all_labels)
    return latent, all_labels


def get_latent_loader(ae, loader, batch_size=64, train=True):
    latent, all_labels = get_latent_data(ae, loader, train)
    data = TensorDataset(latent, all_labels)
    loader = DataLoader(data, batch_size=batch_size, drop_last=train, shuffle=train)
    return loader

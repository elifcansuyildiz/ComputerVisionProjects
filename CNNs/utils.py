"""
Elif Cansu YILDIZ 06/2021
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, optimizer, loss_function, device, debug=False):
    
    total_loss = 0
    total_accuracy = 0
    
    for index, (trainingImages,trainingLabels) in enumerate(dataloader):
        optimizer.zero_grad()

        trainingImages = trainingImages.to(device)
        trainingLabels = trainingLabels.to(device)

        prediction = model(trainingImages)
        
        loss = loss_function(prediction, trainingLabels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
                   
        #Accuracy
        pred_label = torch.argmax(prediction, dim=1)
        accuracy = ((pred_label == trainingLabels).sum().item()) / len(trainingImages)
        total_accuracy += accuracy
        
        if debug:
            with torch.no_grad():
                if index % 100 == 0:
                    print('Train {}/{} Loss {:.6f} Accuracy {:.6f}'.format(index, len(dataloader), loss, accuracy))
        
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
        
    return avg_loss, avg_accuracy

@torch.no_grad()
def evaluate(model, dataloader, loss_function, device):

    total_loss = 0
    total_accuracy = 0
    
    for (test_images, test_labels) in dataloader:

        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        #Loss
        prediction = model(test_images)
        loss = loss_function(prediction, test_labels).item()
        total_loss += loss

        #Accuracy
        pred_label = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(pred_label == test_labels.detach().cpu().numpy())
        total_accuracy += accuracy

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy

def visualize_loss_acc(models_losses, models_accuracies, title=""):
    
    nepochs = 10
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 5))

    ax1.plot(models_losses[0:1*nepochs], label="Shallow Model Loss")
    ax1.plot(models_losses[1*nepochs:2*nepochs], label="Wider Model Loss")
    ax1.plot(models_losses[2*nepochs:3*nepochs], label="Deeper Model (BatchNorm=False) Loss")
    ax1.plot(models_losses[3*nepochs:4*nepochs], label="Deeper Model (BatchNorm=True) Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid()
    ax1.legend()

    ax2.plot(models_accuracies[0:1*nepochs], label="Shallow Model Acc")
    ax2.plot(models_accuracies[1*nepochs:2*nepochs], label="Wider Model Acc")
    ax2.plot(models_accuracies[2*nepochs:3*nepochs], label="Deeper Model (BatchNorm=False) Acc")
    ax2.plot(models_accuracies[3*nepochs:4*nepochs], label="Deeper Model (BatchNorm=True) Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid()
    ax2.legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
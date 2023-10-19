
import torch
from utils import *
from train_q2 import ResNet
from voc_dataset import VOCDataset
import matplotlib.pyplot as plt
from matplotlib import patches

from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.manifold import TSNE

def plot_tsne(projected,targets):
    class_names = VOCDataset.CLASS_NAMES
    colors_array = np.array([[np.random.choice(np.arange(256), size=3)] for i in range(20)])
    color_mean = []
    for i in range(projected.shape[0]):
        colors_point = colors_array[np.where(targets[i, :]==1)]
        mean_colors =  np.mean(colors_point, axis=0, dtype=np.int64)
        color_mean.append(mean_colors)
    
    
    plt.figure(figsize=(10,10))
    plt.scatter(projected[:, 0], projected[:, 1], c=np.array(color_mean)/255)
    plt.legend(handles=[patches.Patch(color=np.array(colors_array[i])/255, label=str(class_names[i])) for i in range(20)])
    plt.title("tsne_plot")
    plt.savefig("tsne.png")
    
def main():
    class_names = VOCDataset.CLASS_NAMES
    device = 'cuda'
    ckpt = '/home/arpitsah/Desktop/Fall-2023/VLR/hw1/q1_q2_classification/checkpoint-model-epoch50.pth'

    model =  ResNet(num_classes=20).cuda()
    model = torch.load(ckpt)
    model = torch.nn.Sequential(*list(model.resnet.children())[:-1])
    model.eval()
    
    data_loader = get_data_loader('voc', train=False, batch_size=50, split='test')
    
    features =[]
    target_labels =[]
    count=0
    for img,label,_  in data_loader:
        count+=1
        img = img.to(device)
        label = label.view(-1,20).to(device)
        extracted_feature = model(img)
        extracted_feature = extracted_feature.reshape(img.shape[0],-1)
        features.append(extracted_feature.detach().cpu().numpy())
        target_labels.append(label.detach().cpu().numpy())
        if count ==20:
            break
    features = np.concatenate(features)
    target_labels = np.concatenate(target_labels)  
    tsne = TSNE()
    projected = tsne.fit_transform(features)
    
    plot_tsne(projected,target_labels)

if  __name__ == '__main__':
    main()
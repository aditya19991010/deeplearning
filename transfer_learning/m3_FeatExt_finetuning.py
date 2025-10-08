## Model 3  - Feature extraction and fine tuning

print('Importing libraries')
import torch
from m1_m2_res18_to_cifar10_p1 import training_loop
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from m1_m2_res18_to_cifar10_p1 import *

os.environ["HTTP_PROXY"] = "http://249000049%40ibab.ac.in:Aditya0%40@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://249000049%40ibab.ac.in:Aditya0%40@proxy.ibab.ac.in:3128"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device ='cpu'
torch.manual_seed(42)

print('Loading checkpoint data')
f= torch.load('DL_model_checkpoint_1756988712', weights_only=False)

# Initiating model
model = models.resnet18(weights=None)

# Adding a linear layer in the end
model.fc = nn.Linear(model.fc.in_features, 10)

# loading fine-tuned model states of layer in the updated model
model.load_state_dict(f['model_state_dict'])
model.eval()


# Feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])

batch = 64

trans_manual = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_manual)
train_load = DataLoader(trainset, batch_size=batch, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_manual)
test_load = DataLoader(testset, batch_size=batch, shuffle=False)

features, labels = [], []

for x, y in train_load:
    with torch.no_grad():
        ft = feature_extractor(x)
    features.append(ft)
    labels.append(y)


#concatenating features
features = torch.cat(features, dim=0)
labels = torch.cat(labels, dim=0)

#Saving feature information
torch.save({'features': features, 'labels': labels}, "CIFAR10_features.pth")

# Creating a classifer
class Classifier(nn.Module):
    def __init__(self, ip =512, nclass=10):
        super().__init__()
        self.fc = nn.Linear(ip,nclass)

    def forward(self,x):
        return self.fc(x)

# loading feature extracted data
data = torch.load(f='CIFAR10_features.pth')


X = data['features']
X = X.view(-1, 512) # to get the [50000, 512] dimension

y = data['labels']

#Create a tensor dataset
dataset = TensorDataset(X,y)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

# Set up params
epoch = 100
lr = 0.001
loss_fn = nn.CrossEntropyLoss()

clf = Classifier(ip=X.shape[1], nclass=len(torch.unique(y)))
optim = torch.optim.SGD( clf.parameters(),lr=lr, momentum=0.9)


## start training
for i in range(epoch):
    total_loss = 0
    ave_loss, accuracy = training_loop(dataloader, loss_fn, optim, clf)
    print(f"Epoch {i+1}; loss = {ave_loss}; acc = {accuracy}")




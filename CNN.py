import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,batchNorm=True):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.drop_out = nn.Dropout()
        # 2x3 : 576, 1000
        self.fc1 = nn.Linear(1600, 1000)
        self.fc2 = nn.Linear(1000, 27)

        #self.batchNorm = batchNorm
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        #self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        #self.conv3 = nn.Conv2d(32,64, kernel_size=3)

        #self.fc1 = nn.Linear(in_features = 64 * 3 * 3, out_features = 150)
        #self.fc2 = nn.Linear(in_features = 150,out_features =  90)
        #self.fc3 = nn.Linear(in_features = 90,out_features = 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

if __name__ == '__main__':
    print("hi")

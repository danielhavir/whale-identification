import torch
import torch.nn as nn

class SiameseWrapper(nn.Module):
    def __init__(self, model, distance_fn=None):
        super(SiameseWrapper, self).__init__()
        self.model = model
        self.distance_fn = distance_fn

    def forward(self, inputs):
        output1 = self.model(inputs[:,0])
        output2 = self.model(inputs[:,1])

        if self.distance_fn is not None:
            output = self.distance_fn(output1, output2)
        else:
            output = (output1, output2)
        
        return output

class TripletWrapper(nn.Module):
    def __init__(self, model, distance_fn=None):
        super(TripletWrapper, self).__init__()
        self.model = model

    def forward(self, inputs):
        anchor = self.model(inputs[:,0])
        positive = self.model(inputs[:,1])
        negative = self.model(inputs[:,2])

        return anchor, positive, negative

class TripletHead(nn.Module):
    def __init__(self, model, distance_fn=None):
        super(TripletHead, self).__init__()
        self.model = model
        #for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(self.model.fc.in_features, 32, (4,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(32, 1, (1,32))
        self.sum = nn.Linear(512, 1)

    def features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.max_pool(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def head(self, input1, input2):
        x1 = input1.mul(input2)
        x2 = input2.add(input2)
        x3 = (input1.sub(input2)).abs()
        x4 = x3.pow(2)

        inputs = torch.stack([x1, x2, x3, x4], dim=0).unsqueeze(-1)
        inputs = self.relu(self.conv1(inputs)).view(x1.size(0), x1.size(1), 32, 1)
        inputs = self.conv2(inputs).view(x1.size(0), -1)
        inputs = self.sum(inputs)

        return inputs

    def forward(self, inputs):
        anc = self.features(inputs[:,0])
        pos = self.features(inputs[:,1])
        neg = self.features(inputs[:,2])

        positive = self.head(anc, pos)
        negative = self.head(anc, neg)

        return positive, negative

def similarity_matrix(mat):
    r = torch.mm(mat, mat.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    D = diag + diag.t() - 2*r
    return D.sqrt()

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

def similarity_matrix(mat):
    r = torch.mm(mat, mat.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    D = diag + diag.t() - 2*r
    return D.sqrt()

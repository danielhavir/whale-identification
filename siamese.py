import torch.nn as nn

class SiameseWrapper(nn.Module):
    def __init__(self, model, distance_fn=None):
        super(SiameseWrapper, self).__init__()
        self.model = model
        self.distance_fn = distance_fn
    
    def predict(self, image, batch):
        if batch.size(0) < image.size(0):
            image = image[:batch.size(0)]

        output1 = self.model(image)
        output2 = self.model(batch)

        if self.distance_fn is not None:
            output = self.distance_fn(output1, output2)
        else:
            output = (output1, output2)
        
        return output

    def forward(self, inputs):
        output1 = self.model(inputs[:,0])
        output2 = self.model(inputs[:,1])

        if self.distance_fn is not None:
            output = self.distance_fn(output1, output2)
        else:
            output = (output1, output2)
        
        return output

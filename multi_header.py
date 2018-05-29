import random
import json
import os
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.autograd import Variable
from torch.nn import functional

class LineSegmentDetector(torch.nn.Module):
    def __init__(self):
        super(LineSegmentDetector, self).__init__()
        self.width = 4
        self.filename = 'multi-header.pkl'
        self.define_layers()
    
    def define_layers(self):
        n_feature, n_hidden, n_output = self.width**2, 2*self.width**2, self.width
        self.layer1 = torch.nn.Linear(n_feature, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)
        self.clfHeader = torch.nn.Linear(n_output, 2)
        self.regressionHeader = torch.nn.Linear(n_output, 1)

    def forward(self, x):
        
        layerStructure1 = functional.tanh(self.layer1(x))
        #layerStructure1 = functional.relu(self.layer1(x))
        layerStructure2 = functional.relu(self.layer2(layerStructure1))
        #layerStructure2 = functional.relu(self.layer2(layerStructure1)) 
        self.clfHeaderLayer = self.clfHeader(layerStructure2)
        self.regressionHeaderLayer = self.regressionHeader(layerStructure2)
        return self.clfHeaderLayer, self.regressionHeaderLayer 

    def draw_line(self):
        width = self.width
        im = Image.new('L', (width, width), 255)
        draw = ImageDraw.Draw(im)
        
        p = random.randint(0,width-1)
        direction = random.randint(0,1)
        if direction == 0:
            draw.line([(p, 0),(p, width-1)], fill=0)
            y1 = torch.squeeze(torch.from_numpy(np.array([1, 0])))
        else:
            draw.line([(0,p),(width-1, p)], fill=0)
            y1 = torch.squeeze(torch.from_numpy(np.array([0, 1])))
            
        x = np.array(im, dtype=np.float32).reshape([width*width]) / 255.
        x = torch.squeeze(torch.from_numpy(x))
        
        y2 = torch.from_numpy(np.array([float(p)/float(width-1)], dtype=np.float32))
        
        return Variable(x), Variable(y1), Variable(y2)


    def train(self):
        if os.path.exists(self.filename):
            self.restore()
            return
        
        print("train")
        print(self)
        losses1 = []
        losses2 = []
        optimizer = torch.optim.Adam(self.parameters()) #, lr= 1e-4
        for epoch in range(100000):
            x, y1, y2 = self.draw_line()
            #x = torch.unsqueeze(x, dim=0)            
            p1, p2 = self(x)
            
            p1 = torch.unsqueeze(p1, dim=0)
            y1 = torch.unsqueeze(y1, dim=0)
            
            p = np.zeros(2, dtype=np.float64)
            p[np.argmax(p1.data.numpy())] = 1
            
            y1 = torch.unsqueeze(torch.argmax(y1), dim=0)
            
            loss1 = torch.nn.CrossEntropyLoss()(p1, y1)
            loss2 = torch.nn.MSELoss()(p2, y2)
            loss = loss1 + loss2
            losses1.append(loss1.data.numpy())
            losses2.append(loss2.data.numpy())
            if len(losses1) > 1000:
                losses1 = losses1[1:]
                losses2 = losses2[1:]
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 1000 == 0:
                loss1 = np.mean(losses1)
                loss2 = np.mean(losses2)
                print(epoch, loss1, loss2, loss1+loss2) 
        
        torch.save(self.state_dict(), self.filename)
        print("done")
    
    def restore(self):
        print("restore")
        return self.load_state_dict(torch.load(self.filename))
        
    
    def predict(self):
        x, y1, y2 = self.draw_line()
        p1, p2 = self(x)
        x, y1, y2 = x.data.numpy(), y1.data.numpy(), y2.data.numpy()
        p1, p2 = p1.data.numpy(), p2.data.numpy()
        
        p20 = p2.copy()
        p = np.zeros(2, dtype=np.long)
        p[np.argmax(p1)] = 1.
        p1 = p
        
        p2[0] = round(p2[0] * float(self.width-1), 0)/float(self.width-1)
        
        print(y1.tolist(), p1.tolist(), p2.tolist(), y2.tolist())

        if (p1.tolist() == y1.tolist() and p2.tolist() == y2.tolist()):
            return True
        return False
    
    def evaluate(self):
        correctCount = 0
        for i in range(1000):
            if self.predict():
                correctCount+=1
        print(correctCount/10.) 
lsd = LineSegmentDetector()
lsd.train()
lsd.evaluate()


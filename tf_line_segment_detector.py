# -*- coding: UTF-8 -*-
import random
import os
import numpy as np
import PIL
from PIL.ImageFilter import BuiltinFilter
from PIL import Image, ImageDraw

import tensorflow as tf

from loss_monitor import LossMonitor

class TfLineSegmentDetector:

    VERT_DIR = 0
    HORIZ_DIR = 1
    NOT_FOUND = 2
    
    def __init__(self, width=4, forceTraining=False):
        
        self.width = width
        self.forceTraining = forceTraining
        # self.epoch = 20000
        self.epoch = 100000
        self.reportEpoch = 1000
        self.numClassification = 3
        self.version = '0.2.4'
        self.filename = 'summary/line-segment-detector.ckpt'#'line-segment-detector.pkl'
        self.lossMonitor = True
        self.lossMonitorLength = 30
                
        self.define_placeholders()
        self.define_layers()
        self.define_losses()
        
        self.initializer = tf.global_variables_initializer()
        self.mergedSummary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        
        print('TfLineSegmentDetector(version:%s, width:%d, forceTraining:%s)' % (
            self.version,
            self.width,
            str(self.forceTraining)))
    
    def define_placeholders(self):
        numFeature = self.width**2
        self.X = tf.placeholder(shape=[None, numFeature], dtype=tf.float32)
        self.y1 = tf.placeholder(shape=[None, self.numClassification], dtype=tf.float32, name="y1")
        self.y2 = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y2")
    
    def define_layers(self):
        numHidden, numOutput = 2*self.width**2, self.width
        self.layer1 = tf.layers.dense(self.X, numHidden, activation=tf.nn.tanh, name="layer1")
        self.layer2 = tf.layers.dense(self.layer1, numOutput, activation=tf.nn.relu, name="layer2")
        
        self.clfHeader = tf.layers.dense(self.layer2, self.numClassification, name="clfHeader")
        self.regressionHeader = tf.layers.dense(self.layer2, 1, name="regressionHeader")
        
    
    def define_losses(self):
        self.loss1 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.clfHeader,
                labels=self.y1,
                name = "loss1"
            )
        )
        self.loss2 = tf.reduce_mean(
            tf.square(
                self.y2 - self.regressionHeader
            ), name="loss2")
        self.loss = self.loss1 + self.loss2
        
        tf.summary.scalar("loss1", self.loss1)
        tf.summary.scalar("loss2", self.loss2)
        tf.summary.scalar("loss", self.loss)
        
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def draw_line(self, noisePercentage=0.1):
        bgcolor, forecolor = 0, 255# random.randint(128, 255)
        im = Image.new('L', (self.width, self.width), bgcolor)
        draw = ImageDraw.Draw(im)
        
        pixels = im.load()
        noisePointNum = max(min(int(self.width ** 2 * noisePercentage), int(self.width ** 2 * 0.5)), 0)
        noisePointNum = random.randint(0, noisePointNum)
        for i in range(noisePointNum):
            noiseX, noiseY, noiseColor = random.randint(0, self.width-1), random.randint(0, self.width-1), random.randint(0, 255)
            pixels[noiseX, noiseY] = noiseColor
        
        p = random.randint(0, self.width-1)
        direction = random.randint(0, self.numClassification-1)
        if direction == 0:
            draw.line([(p, 0),(p, self.width-1)], fill=forecolor)
        elif direction == 1:
            draw.line([(0,p),(self.width-1, p)], fill=forecolor)
        else:
            # make sure, when blank image, it will never predict anything
            # so always set it to 0.0
            p = 0.0

        x = np.array(im, dtype=np.float32).reshape([1, self.width*self.width]) / 255.
    
        y1 = np.zeros(self.numClassification, dtype=np.float32)
        y1[direction] = 1.0
        y1 = y1.reshape([1, self.numClassification])
        y2 = np.array([float(p)/float(self.width-1)], dtype=np.float32).reshape([1, 1])
        
        return x, y1, y2


    def train(self):
        if not self.forceTraining and os.path.exists(self.filename+".index"):
            self.restore()
            return
        

        print("training started")
        losses1 = []
        losses2 = []
        
        lossMonitor = LossMonitor(lossMonitorLength = self.lossMonitorLength, switchOn=self.lossMonitor)
        print("epoch\tclassifierLoss\tregressionLoss\ttotalLoss")
        
        self.sess = tf.Session()
        self.sess.run(self.initializer)
        writer = tf.summary.FileWriter("summary", self.sess.graph)
    
        for epoch in range(self.epoch):
            x, y1, y2 = self.draw_line()

            _, loss, loss1, loss2, summary = self.sess.run([
                    self.optimizer,
                    self.loss, 
                    self.loss1,
                    self.loss2,
                    self.mergedSummary],
                feed_dict={
                    self.X: x, 
                    self.y1: y1,
                    self.y2: y2}
            )
            
            writer.add_summary(summary, epoch)
            losses1.append(loss1)
            losses2.append(loss2)
            if len(losses1) > 1000:
                losses1 = losses1[1:]
                losses2 = losses2[1:]
            
            if epoch % self.reportEpoch == 0:
                loss1 = np.mean(losses1)
                loss2 = np.mean(losses2)
                print("%d\t%f\t%f\t%f" % (epoch, loss1, loss2, loss), end='')
                print(lossMonitor.monitor(loss))

                
            if epoch % (self.reportEpoch*10) == 0:
                self.evaluate()
                pass
        
        self.saver.save(self.sess, self.filename)
        print("training finished")
    
    def __del__(self):
        if 'sess' in dir(self) and self.sess != None:
            self.sess.close()
            
    def restore(self):
        print("restore")
        self.sess = tf.Session()
        
        self.sess.run(self.initializer)
        self.saver.restore(self.sess, self.filename)
    
    def predict(self):
        x, y1, y2 = self.draw_line()
        
        p1, p2 = self.sess.run(
            [self.clfHeader, self.regressionHeader],
            feed_dict={
                self.X:x
            }
        )
        
        
        p1 = np.argmax(p1)
        tmp = np.zeros(self.numClassification)
        tmp[p1] = 1.0
        p1 = tmp
        p2[0][0] = round(p2[0][0] * float(self.width-1), 0)/float(self.width-1)
        
        if (p1.tolist() == y1[0].tolist() and p2.tolist() == y2.tolist()):
            return True
        return False
    
    def _predict_im(self, im, x, y):
        cropRegion = (x, y, x+self.width, y+self.width)
        croppedIm = im.crop(cropRegion)
        data = np.array(croppedIm, dtype=np.float32).reshape([1, self.width*self.width]) / 255.
        
        p1, p2 = self.sess.run([self.clfHeader, self.regressionHeader], feed_dict={self.X:data})
        
        p1 = np.argmax(p1)

        p2 = round(p2[0][0] * float(self.width-1), 0)/float(self.width-1)        
        p2 = int(round(p2 * float(self.width-1),5))

        return x, y, p1, p2
    
    def detect(self, im, showImage=False):
        if showImage:
            im2 = im.copy()
            draw = ImageDraw.Draw(im2)
        im = im.convert('L')
        
        
        im = im.filter(PIL.ImageFilter.CONTOUR)
        horzIm, vertIm = im.filter(horzEdgeFilter), im.filter(vertEdgeFilter)
        #vertIm.show()
        lineThreshold = int(float(min(horzIm.size[0],horzIm.size[1])) * 0.2)
        
        result = []
        # horizontal line
        for y in range(horzIm.size[1]//self.width):
            prevX, prevY = -1, -1
            # start, end, length, direction
            currLine = [-1, -1, -1, 1]
            for x in range(horzIm.size[0]//self.width):
                _, _, direction, pos = lsd._predict_im(horzIm, x*self.width, y*self.width)
                #if direction == self.1 and pos>=0:
                if direction == TfLineSegmentDetector.HORIZ_DIR and pos>=0:
                
                    #print(x,y,pos,direction)
                    #draw.line([(x*self.width,y*self.width+pos),(x*self.width+self.width-1, y*self.width+pos)], fill=(255,0,0))
                    if pos == prevY and x-1 == prevX:
                        currLine[2] += self.width
                        #print('length++')
                    else:
                        if currLine[2] >= lineThreshold:
                            #print('appendLine')
                            result.append(currLine)
                        currLine = [x*self.width, y*self.width+pos, self.width, direction]
                        #print('newLine')
                prevX, prevY = x, pos
            if currLine[2] >= lineThreshold:
                result.append(currLine)
                #print('appendLine')
        
        # vertical line
        for x in range(vertIm.size[0]//self.width):
            prevX, prevY = -1, -1
            # start, end, length, direction
            currLine = [-1, -1, -1, 0]
            for y in range(vertIm.size[1]//self.width):            
                _, _, direction, pos = lsd._predict_im(vertIm, x*self.width, y*self.width)
                #if direction == 0 and pos>=0:
                if direction == TfLineSegmentDetector.VERT_DIR and pos>=0:
                    #draw.line([(x*self.width+pos, y*self.width),(x*self.width+pos, y*self.width+self.width-1)], fill=(255,0,0))
                    if pos == prevX and y-1 == prevY:
                        currLine[2] += self.width
                        #print('length++')
                    else:
                        if currLine[2] >= lineThreshold:
                            #print('appendLine')
                            result.append(currLine)
                        currLine = [x*self.width+pos, y*self.width, self.width, direction]
                        #print('newLine')
                prevX, prevY = pos, y


            if currLine[2] >= lineThreshold:
                result.append(currLine)
        
        res = []        
        for line in result:
            if line[3] == 1:
                if showImage:
                    draw.line([(line[0],line[1]),(line[0]+line[2], line[1])], fill=(255,0,0))
                res.append([(line[0],line[1]),(line[0]+line[2], line[1])])
            else:
                if showImage:
                    draw.line([(line[0],line[1]),(line[0], line[1]+line[2])], fill=(255,0,0))
                res.append([(line[0],line[1]),(line[0], line[1]+line[2])])

        
        if showImage:
            im2.show()
        return res
    
    def evaluate(self):
        correctCount = 0
        for i in range(1000):
            if self.predict():
                correctCount+=1
        correctPercentage = float(correctCount)/10.
        print("correctPercentage: %f%s" % (correctPercentage, "%"))
        # correctPercentage: 97.100000%

class horzEdgeFilter(BuiltinFilter):
    name = "Horizontal Edges"
    filterargs = (3, 3), 1, 0, (
         1,  1,  1,
         0,  0,  0,
        -1, -1, -1
        )

class vertEdgeFilter(BuiltinFilter):
    name = "Vertical Edges"
    filterargs = (3, 3), 1, 0, (
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
        )

if __name__ == "__main__":
    import datetime
    lsd = TfLineSegmentDetector(10, False)
    lsd.train()
    
    
    st = datetime.datetime.now()
    im = Image.open('img/wx01.png')
    print(lsd.detect(im, True))
    
    #im = Image.open('img/wx02.png')
    #print(lsd.detect(im, True))
    
    #im = Image.open('img/chrome01.png')
    #print(lsd.detect(im, True))
    
    ed = datetime.datetime.now()
    print(ed-st)
    

    
    

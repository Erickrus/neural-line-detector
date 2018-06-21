import numpy as np
class LossMonitor:
    def __init__(self, lossMonitorLength=10, switchOn = True):
        self.lossMonitorLength = lossMonitorLength
        self.monitorCount = 0
        self.minLossPosition = -1
        self.minLoss = 999999.
        self.switchOn = switchOn
        self.lossDelta = []
    
    def monitor(self, loss):
        result = ''
        exceptionText = """Loss doesn't get lower in %d cycles.
Perhaps, this is a wrong model or hyper-parameter should be adjusted."""
        if loss < self.minLoss:
            if len(self.lossDelta) == 0:
                self.lossDelta.append(1)
            else:
                self.lossDelta.append(self.monitorCount-self.minLossPosition)
            self.minLossPosition = self.monitorCount
            self.minLoss = loss
            result = '*'
            
        
        self.monitorCount += 1
        
        #if self.minLossPosition + self.lossMonitorLength < self.monitorCount and self.switchOn:
        #    raise Exception(exceptionText % self.lossMonitorLength)
        if len(self.lossDelta) > 3:
            self.lossDelta = self.lossDelta[1:]
        deltaLen = max(int(np.mean(self.lossDelta) * 2.5), self.lossMonitorLength)
        #print('\t', int(np.mean(self.lossDelta) * 2.), deltaLen, end='')
        if self.switchOn and self.minLossPosition + deltaLen < self.monitorCount:
            raise Exception(exceptionText)
        return result

        
    
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.nn import global_add_pool as gap, global_mean_pool as gmp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.nn import global_add_pool as gap, global_mean_pool as gmp
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,in_channels):
        super(Net,self).__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.pool1 = TopKPooling(32, ratio = 0.9)
        self.conv2 = GCNConv(32,64)
        self.pool2 = TopKPooling(64, ratio = 0.9)
        self.conv3 = GCNConv(64, 128)
        self.pool3 = TopKPooling(128, ratio = 0.9)
        self.conv4 = GCNConv(128, 128)
        self.line1 = torch.nn.Linear(128,64)
        self.line2 = torch.nn.Linear(64,64)
        self.line3 = torch.nn.Linear(64,32)
        self.line4 = torch.nn.Linear(32,16)
        self.line5 = torch.nn.Linear(16,1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.bn4 = torch.nn.BatchNorm1d(16)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x,edge_index,_,batch,_,_ = self.pool1(x,edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x,edge_index,_,batch,_,_ = self.pool2(x,edge_index, None, batch)
        x = F.relu(self.conv3(x, edge_index))
        x,edge_index,_,batch,_,_ = self.pool3(x,edge_index, None, batch)
        x = F.relu(self.conv4(x, edge_index))
        x = gmp(x,batch)
        x = self.line1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.line2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.line3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.line4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.line5(x)
        return x
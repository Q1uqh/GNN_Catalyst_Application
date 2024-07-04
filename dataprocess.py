from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import dgl
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops

mol = Chem.MolFromMolFile('/Datasets/AL2O3.mol',sanitize = False)

BeginAtomIdx = []
EndAtomIdx = []
for i in range(mol.GetNumBonds()):
    BeginAtomIdx.append(mol.GetBondWithIdx(i).GetBeginAtomIdx())
    EndAtomIdx.append(mol.GetBondWithIdx(i).GetEndAtomIdx())
chemical_bond = dgl.graph((BeginAtomIdx, EndAtomIdx), num_nodes=7)
topological_structure = dgl.to_bidirected(chemical_bond)
print(topological_structure.edges())

atomic_features = pd.read_excel('/Datasets/node_feat.xlsx',index_col = 0,header = 1)
atomic_features.drop(['atom'], axis = 1,inplace = True)

config_feat = pd.read_excel('/Datasets/data.xlsx',index_col = 0)
label = config_feat['E0']
config_feat.drop(['E0'],inplace = True,axis = 1)

config_feat_train, config_feat_test, label_train, label_test = train_test_split(config_feat, label, test_size = 0.05, random_state = 0)

class makedataset(InMemoryDataset):
    def __init__(self, root, figure = None, atom_features = None, edge = None, label = None, transform = None, pre_transform = None):
        self.figure = figure
        self.atom_features = atom_features
        self.edge = edge
        self.label = label
        super(makedataset, self).__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['gnn_dataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        for j in self.figure.index:
            LS = []
            for i in self.figure.loc[j]:
                ls = self.atom_features.iloc[int(i), :].values
                LS.append(ls)
            node_features = pd.DataFrame(LS).values
            y = torch.tensor(label[j], dtype = torch.float32)
            x = torch.tensor(node_features, dtype = torch.float32)
            edge_index = torch.stack(self.edge).long()
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

makedataset(root = '/output/train_data/', figure = config_feat_train, atom_features = atomic_features, edge = topological_structure.edges(), label = label_train)
        
makedataset(root = '/output/test_data/', figure = config_feat_test, atom_features = atomic_features, edge = topological_structure.edges(), label = label_test)
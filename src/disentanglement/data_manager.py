import pickle
import random
from torch.utils import data

class DtwDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data_dict = self.data[index] #["{emo_list[j]}_{level}_{filename}"]
        data_list = list(data_dict)
        key1, key2 = random.sample(data_list, 2)
        vtx_e1, vtx_e2 = data_dict[key1], data_dict[key2]

        # different contents
        half_fr = int(len(vtx_e1)/2)
        vtx_c1e1, vtx_c2e1 = vtx_e1[:half_fr], vtx_e1[half_fr:2*half_fr]
        vtx_c1e2, vtx_c2e2 = vtx_e2[:half_fr], vtx_e2[half_fr:2*half_fr]

        return vtx_c1e1, vtx_c2e1, vtx_c1e2, vtx_c2e2

    def __len__(self):
        return len(self.data)

def read_data(hparams):
    print("Loading data...")
    f = open(hparams.vtx_dtw_path, 'rb')
    vtx = pickle.load(f)
    f.close()

    train_data, valid_data, test_data = [], [], []
    for k_sentence_num, v_dict in vtx.items():
        if int(k_sentence_num) == 12:
            valid_data.append(v_dict)
            test_data.append(v_dict)
        else:
            train_data.append(v_dict)

    print(f"[Data length] Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    return train_data, valid_data, test_data

def get_dataloader(hparams):
    dataset = {}
    train_data, valid_data, test_data = read_data(hparams)
    train_data = DtwDataset(train_data)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=hparams.batch_size, shuffle=True)
    valid_data = DtwDataset(valid_data)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=hparams.batch_size, shuffle=False)
    test_data = DtwDataset(test_data)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=hparams.batch_size, shuffle=False)
    return dataset

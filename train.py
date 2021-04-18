from dataset import EntailmentDataset
from torch.utils.data import DataLoader
from model import RoBERTaNLI
from torch import nn
from torch import optim

def train(epochs, batch_size, data, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(0,epochs):
        model.train()
        for i, input_data in enumerate(data):
            print('Batch', i)
            optimizer.zero_grad()
            print('input ids', input_data['input_ids'].size())
            print('attention_mask', input_data['attention_mask'].size())
            print('token type ids', input_data['token_type_ids'].size())
            output = model(input_data['input_ids'], input_data['attention_mask'], input_data['token_type_ids'])
            loss = criterion(output, labels)
        print({"Epoch": epoch, "Loss": loss.item()})
    return
if __name__ == "__main__":
    batch_size=100
    epochs=1
    dataset  = EntailmentDataset("~/Datasets/contradictory-watson/train.csv", max_length=259)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model = RoBERTaNLI()
    train(epochs, batch_size, data_loader, model)

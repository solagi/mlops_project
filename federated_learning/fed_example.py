import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD


# Load and transform the MNIST dataset
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(f"Number of training samples: {len(train_data)}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
    

class Server:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.device = device
        self.model = model.to(self.device)

    def aggregate(self, deltas: dict [str, torch.Tensor]) -> None:
        """
        Update the global model with clients' deltas. 
        """
        new_state_dict = copy.deepcopy(self.model.state_dict())
        for key in new_state_dict:
            # each server sends in new deltas, we average them (sum / len) and add to the global model
            new_state_dict[key] += sum(delta[key] for delta in deltas) / len(deltas)
        self.model.load_state_dict(new_state_dict)
        print("Global model updated with aggregated deltas.")

    
class Client:
    # client does three things: training, testing and updating
    def __init__(self, train_data: Dataset, test_data: Dataset, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = DataLoader(train_data, batch_size=50, shuffle=True) # shuffels the data locally
        self.test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    def train(self, epochs: int = 1) -> tuple[dict[str, nn.Tensor], float]:
        """
        Train the local client model on the local data. Return the model's delta and the final loss."""
        initial_state = copy.deepcopy(self.model.state_dict())
        self.model.train()
        total_loss = 0.0
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                self.optimizer.step() #the optimizer has the moment, hense its been updated
                total_loss += loss.item()
        delta = copy.deepcopy(self.model.state_dict())

        for key in delta:
            delta[key] -= initial_state[key]

        return delta, total_loss
    
    def test(self) -> tuple[float, float]:
        """
        Test the local model on the local data. Return the test loss annd the model accuracy.
        """
        self.model.eval()
        test_loss = 0.0
        test_size = len(self.test_loader.dataset)
        correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_loss += F.nll_loss(output, y, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= test_size
        accuracy = 100* correct / test_size

        return test_loss, accuracy
    
    def update(self, model: nn.Module) -> None:
        """
        Updates the client model.
        """
        self.model.load_state_dict(copy.deepcopy(model.state_dict())) # use copy here as pytroch uses pointers


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    server = Server(model=Net(), device=device)

    clients = []
    num_clients = 5

    # Partition the dataset evenly among clients
    train_split = torch.utils.data.random_split(train_data, [len(train_data) // num_clients for _ in range(num_clients)])
    test_split = torch.utils.data.random_split(test_data, [len(test_data) // num_clients for _ in range(num_clients)])

    for i in range(num_clients):
        clients.append(Client(
            train_data=train_split[i],
            test_data=test_split[i],
            model=Net()
            # device=device
        ))
    
    # Run the distribution training process
    rounds = 10
    epochs = 1

    for i in range(rounds):
        print(f"Starting round {i}...")

        deltas, train_losess, test_losses, test_accuracies = [], [], [], []
        for i,client in enumerate(clients):
            delta, train_loss = client.train(epochs=epochs)
            deltas.append(delta)
            train_losess.append(train_loss)
            # test_losses....
            print(f"Client {i} trained with loss: {train_loss:.4f}")
            test_loss, test_accuracy = client.test()

            print(f"Client {i} test loss: {test_loss:.4f}, accuracy: {test_accuracy:.2f}%")
        
        print("Aggregating deltas...")
        models = [client.model for client in clients]
        server.aggregate(deltas)

    # evaluate the global model performance of each client
    for i, client in enumerate(clients):
        client.update(server.model)
        test_loss, test_accuracy = client.test()
        print(f"Client {i} final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.2f}%")
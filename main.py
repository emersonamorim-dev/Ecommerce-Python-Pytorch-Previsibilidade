import torch
import torch.nn as nn

class EcommerceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EcommerceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out, _ = self.lstm1(out)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.fc2(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Carregue seus dados de treinamento
train_data = torch.randn(1000, 1, 10)

# Defina os parâmetros do modelo
input_size = 10  

# 10 recursos
hidden_size = 64  

# número de neurônios na camada oculta
num_layers = 2  

# uma única saída
output_size = 1  


# Crie uma instância do modelo
model = EcommerceModel(input_size, hidden_size, num_layers, output_size)

# Defina a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Treine o modelo
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
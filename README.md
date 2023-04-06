## Modelo de Ecommerce

- Codificação em Python usando Pytorch para implementação de uma rede neural Long Short-Term Memory (LSTM) para um problema hipotético de comércio eletrônico. 
- O modelo recebe um tensor de forma de entrada (batch_size, sequence_length, input_size) e gera um tensor de forma (batch_size, output_size).


- Carrega seus dados de treinamento como um tensor PyTorch de forma (batch_size, sequence_length, input_size)
- Define os parâmetros do modelo: input_size, hidden_size, num_layers, output_size
- Cria uma instância da classe EcommerceModel com os parâmetros definidos
- Define a função de perda e o otimizador
- Treina o modelo para um número especificado de épocas usando o tensor train_data e train_labels
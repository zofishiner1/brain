import numpy as np
import networkx as nx
import plotly.graph_objects as go

class Neuron:
    def __init__(self, number_of_weights=1, neuron_id=None, neuron_type="Input", activation_function="sigmoid"):
        self.w = np.random.normal(size=number_of_weights)  # Случайные веса
        self.b = np.random.normal()  # Случайное смещение
        self.id = int(neuron_id)  # Преобразование идентификатора в int
        self.type = neuron_type  # Тип нейрона (входной, скрытый, выходной)
        self.activation_function = activation_function  # Функция активации
        self.activity = 0  # Активность нейрона

    def activate(self, inputs):
        x = np.dot(self.w, inputs) + self.b  # Взвешенная сумма
        if self.activation_function == "sigmoid":
            output = 1 / (1 + np.exp(-x))  # Сигмоидная функция активации
        elif self.activation_function == "relu":
            output = np.maximum(0, x)  # ReLU
        elif self.activation_function == "tanh":
            output = np.tanh(x)  # Гиперболический тангенс
        else:
            raise ValueError("Unknown activation function")
        self.activity = output  # Обновляем активность нейрона
        return output

    def set_weights(self, weights):
        self.w = weights

    def set_bias(self, bias):
        self.b = bias

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_sizes, output_size, learning_rate=0.01,
                 neuron_addition_threshold=0.9, neuron_removal_threshold=0.1,
                 connection_threshold=0.2):
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes  # Список размеров скрытых слоев
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.neuron_addition_threshold = neuron_addition_threshold
        self.neuron_removal_threshold = neuron_removal_threshold
        self.connection_threshold = connection_threshold

        self.input_neurons = [Neuron(neuron_id=i, neuron_type="Input") for i in range(input_size)]

        # Создаем скрытые слои
        self.hidden_layers = []
        self.hidden_neurons = []
        neuron_id_counter = input_size
        for layer_index, layer_size in enumerate(hidden_layers_sizes):
            hidden_layer = [Neuron(neuron_id=neuron_id_counter + i, neuron_type="Hidden") for i in range(layer_size)]
            self.hidden_layers.append(hidden_layer)
            self.hidden_neurons.extend(hidden_layer)  # Keep track of all hidden neurons
            neuron_id_counter += layer_size

        self.output_neurons = [Neuron(neuron_id=neuron_id_counter + i, neuron_type="Output") for i in range(output_size)]

        # Создаем соединения между слоями
        self.input_hidden_connections = self.create_random_connections(input_size, hidden_layers_sizes[0]) if hidden_layers_sizes else []
        self.hidden_hidden_connections = []
        for i in range(len(hidden_layers_sizes) - 1):
            self.hidden_hidden_connections.append(self.create_random_connections(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
        self.hidden_output_connections = self.create_random_connections(hidden_layers_sizes[-1], output_size) if hidden_layers_sizes else self.create_random_connections(input_size, output_size)

        # Инициализируем веса
        self.initialize_weights()

    def create_random_connections(self, source_size, target_size):
        connections = []
        for _ in range(target_size):
            # Каждый нейрон целевого слоя случайно подключается к нейронам исходного слоя
            connected_sources = np.random.choice(
                range(source_size),
                size=np.random.randint(1, source_size + 1),
                replace=False
            )
            connections.append(connected_sources)
        return connections

    def initialize_weights(self):
        # Веса для связей между входным и первым скрытым слоем
        if self.hidden_layers:
            for i, neuron in enumerate(self.hidden_layers[0]):
                self.hidden_neurons[i].w = np.random.normal(size=len(self.input_hidden_connections[i]))

            # Веса для связей между скрытыми слоями
            for layer_index in range(len(self.hidden_layers) - 1):
                for i, neuron in enumerate(self.hidden_layers[layer_index+1]):
                    self.hidden_neurons[sum(self.hidden_layers_sizes[:layer_index+1]) + i].w = np.random.normal(size=len(self.hidden_hidden_connections[layer_index][i]))

            # Веса для связей между последним скрытым слоем и выходным слоем
            for i, neuron in enumerate(self.output_neurons):
                self.output_neurons[i].w = np.random.normal(size=len(self.hidden_output_connections[i]))
        else:
            # Если нет скрытых слоев, соединяем входной слой напрямую с выходным
            self.hidden_output_connections = self.create_random_connections(self.input_size, self.output_size)
            for i, neuron in enumerate(self.output_neurons):
                self.output_neurons[i].w = np.random.normal(size=len(self.hidden_output_connections[i]))

    def predict(self, inputs):
        # Проверяем, что размер входных данных соответствует количеству входных нейронов
        if len(inputs) != len(self.input_neurons):
            raise ValueError("Incorrect number of inputs provided.")

        # Активируем слои последовательно
        layer_outputs = [inputs]  # Начинаем с входных данных

        # Активируем скрытые слои
        for layer_index, hidden_layer in enumerate(self.hidden_layers):
            current_layer_outputs = []
            # Determine connections for the current layer
            if layer_index == 0:
                connections = self.input_hidden_connections
                previous_layer_outputs = layer_outputs[-1]  # Outputs from input layer
            else:
                connections = self.hidden_hidden_connections[layer_index - 1]
                previous_layer_outputs = layer_outputs[-1]  # Outputs from previous hidden layer

            for i, neuron in enumerate(hidden_layer):
                connected_inputs_indices = connections[i]
                # Check if the number of connected inputs matches the number of weights
                if len(connected_inputs_indices) != len(neuron.w):
                    # Adjust weights to match the number of connected inputs
                    num_missing = abs(len(connected_inputs_indices) - len(neuron.w))
                    if len(connected_inputs_indices) > len(neuron.w):
                        neuron.w = np.concatenate((neuron.w, np.random.normal(size=num_missing)))
                    else:
                        neuron.w = neuron.w[:len(connected_inputs_indices)]
                connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices]
                hidden_output = neuron.activate(connected_inputs)
                current_layer_outputs.append(hidden_output)
            layer_outputs.append(current_layer_outputs)

        # Активируем выходные нейроны
        output_layer_outputs = []
        for i, output_neuron in enumerate(self.output_neurons):
            # Determine connections for the output layer
            if self.hidden_layers:
                connections = self.hidden_output_connections
                previous_layer_outputs = layer_outputs[-1]  # Outputs from last hidden layer
            else:
                # If no hidden layers, connect directly to input layer
                connections = self.hidden_output_connections
                previous_layer_outputs = layer_outputs[0]  # Outputs from input layer

            connected_inputs_indices = connections[i]
             # Check if the number of connected inputs matches the number of weights
            if len(connected_inputs_indices) != len(output_neuron.w):
                # Adjust weights to match the number of connected inputs
                num_missing = abs(len(connected_inputs_indices) - len(output_neuron.w))
                if len(connected_inputs_indices) > len(output_neuron.w):
                    output_neuron.w = np.concatenate((output_neuron.w, np.random.normal(size=num_missing)))
                else:
                    output_neuron.w = output_neuron.w[:len(connected_inputs_indices)]
            connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices]
            output_output = output_neuron.activate(connected_inputs)
            output_layer_outputs.append(output_output)

        return output_layer_outputs

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            # Dynamic structure adaptation
            self.adapt_network_structure(inputs, targets)

            # Прямой проход
            layer_outputs = [inputs]

            # Forward pass through hidden layers
            for layer_index, hidden_layer in enumerate(self.hidden_layers):
                current_layer_outputs = []
                if layer_index == 0:
                    connections = self.input_hidden_connections
                    previous_layer_outputs = layer_outputs[-1]
                else:
                    connections = self.hidden_hidden_connections[layer_index - 1]
                    previous_layer_outputs = layer_outputs[-1]

                for i, neuron in enumerate(hidden_layer):
                    # Ensure that the neuron has weights before activation
                    if len(neuron.w) > 0:
                        connected_inputs_indices = connections[i]
                        # Check if the number of connected inputs matches the number of weights
                        if len(connected_inputs_indices) != len(neuron.w):
                            # Adjust weights to match the number of connected inputs
                            num_missing = abs(len(connected_inputs_indices) - len(neuron.w))
                            if len(connected_inputs_indices) > len(neuron.w):
                                neuron.w = np.concatenate((neuron.w, np.random.normal(size=num_missing)))
                            else:
                                neuron.w = neuron.w[:len(connected_inputs_indices)]
                        connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices]
                        hidden_output = neuron.activate(connected_inputs)
                        current_layer_outputs.append(hidden_output)
                    else:
                        # If the neuron has no weights, set its output to 0
                        current_layer_outputs.append(0)  # Or another appropriate default value
                layer_outputs.append(current_layer_outputs)

            # Forward pass through output layer
            output_layer_outputs = []
            if self.hidden_layers:
                connections = self.hidden_output_connections
                previous_layer_outputs = layer_outputs[-1]
            else:
                connections = self.hidden_output_connections
                previous_layer_outputs = layer_outputs[0]

            for i, output_neuron in enumerate(self.output_neurons):
                connected_inputs_indices = connections[i]
                 # Check if the number of connected inputs matches the number of weights
                if len(connected_inputs_indices) != len(output_neuron.w):
                    # Adjust weights to match the number of connected inputs
                    num_missing = abs(len(connected_inputs_indices) - len(output_neuron.w))
                    if len(connected_inputs_indices) > len(output_neuron.w):
                        output_neuron.w = np.concatenate((output_neuron.w, np.random.normal(size=num_missing)))
                    else:
                        output_neuron.w = output_neuron.w[:len(connected_inputs_indices)]
                connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices]
                output_output = output_neuron.activate(connected_inputs)
                output_layer_outputs.append(output_output)

            # Вычисление ошибки
            output_errors = np.array(targets) - np.array(output_layer_outputs)

            # Обратное распространение ошибки
            # Output layer
            for i, output_neuron in enumerate(self.output_neurons):
                connected_inputs_indices = connections[i]
                for j, input_index in enumerate(connected_inputs_indices):
                    delta = output_errors[i] * self.sigmoid_derivative(output_layer_outputs[i])
                    output_neuron.w[j] += self.learning_rate * delta * previous_layer_outputs[input_index]
                output_neuron.b += self.learning_rate * delta

            # Hidden layers (backwards)
            hidden_errors = [np.zeros(len(layer)) for layer in self.hidden_layers]
            for layer_index in reversed(range(len(self.hidden_layers))):
                hidden_layer = self.hidden_layers[layer_index]
                current_layer_outputs = layer_outputs[layer_index + 1]

                if layer_index == len(self.hidden_layers) - 1:
                    # Last hidden layer
                    next_layer_neurons = self.output_neurons
                    next_layer_connections = self.hidden_output_connections
                    next_layer_errors = output_errors
                    next_layer_outputs = output_layer_outputs
                else:
                    # Middle hidden layers
                    next_layer_neurons = self.hidden_layers[layer_index + 1]
                    next_layer_connections = self.hidden_hidden_connections[layer_index]
                    next_layer_errors = hidden_errors[layer_index + 1]
                    next_layer_outputs = layer_outputs[layer_index + 2]

                for i, neuron in enumerate(hidden_layer):
                    error_sum = 0
                    for j, next_neuron in enumerate(next_layer_neurons):
                        # Ensure that the next neuron has connections
                        if j < len(next_layer_connections) and i in next_layer_connections[j]:
                            input_index_in_next_neuron = np.where(next_layer_connections[j] == i)[0][0]
                            delta = next_layer_errors[j] * self.sigmoid_derivative(next_layer_outputs[j])
                            error_sum += delta * next_neuron.w[input_index_in_next_neuron]
                    hidden_errors[layer_index][i] = error_sum

                    # Update weights
                    if layer_index == 0:
                        previous_layer_outputs = layer_outputs[0]
                        connections = self.input_hidden_connections
                    else:
                        previous_layer_outputs = layer_outputs[layer_index]
                        connections = self.hidden_hidden_connections[layer_index - 1]

                    # Check if i is within the bounds of connections
                    if i < len(connections):
                        connected_inputs_indices = connections[i]
                        # Check if connected_inputs_indices is not empty
                        if len(connected_inputs_indices) > 0:
                            # Ensure j is within the bounds of neuron.w
                            original_num_weights = len(neuron.w)
                            num_connections = len(connected_inputs_indices)
                            min_len = min(original_num_weights, num_connections)
                            # Ensure the size of neuron.w matches the number of connections
                            if len(neuron.w) != len(connected_inputs_indices):
                                if len(connected_inputs_indices) > len(neuron.w):
                                    neuron.w = np.concatenate((neuron.w, np.random.normal(size=len(connected_inputs_indices) - len(neuron.w))))
                                else:
                                    neuron.w = neuron.w[:len(connected_inputs_indices)]
                            for j in range(min_len):
                                delta = hidden_errors[layer_index][i] * self.sigmoid_derivative(current_layer_outputs[i])
                                neuron.w[j] += self.learning_rate * delta * previous_layer_outputs[connected_inputs_indices[j]]
                            neuron.b += self.learning_rate * delta

            # Вывод ошибки на каждой эпохе (можно настроить)
            if epoch % 10 == 0:
                mse = np.mean(output_errors ** 2)
                print(f"Epoch {epoch}, MSE: {mse}")

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def add_neuron(self, layer_index):
        """Добавляет новый нейрон в указанный скрытый слой."""
        if 0 <= layer_index < len(self.hidden_layers):
            layer = self.hidden_layers[layer_index]
            new_neuron = Neuron(neuron_id=len(self.hidden_neurons) + len(self.input_neurons), neuron_type="Hidden")
            layer.append(new_neuron)
            self.hidden_neurons.append(new_neuron)
            self.hidden_layers_sizes[layer_index] += 1

            # Update connections
            if layer_index == 0:
                # Connect to input layer
                connected_inputs_indices = np.random.choice(
                    range(len(self.input_neurons)),
                    size=np.random.randint(1, len(self.input_neurons) + 1),
                    replace=False
                )
                self.input_hidden_connections.append(connected_inputs_indices)
                new_neuron.w = np.random.normal(size=len(connected_inputs_indices))
            else:
                # Connect to previous hidden layer
                prev_layer_size = self.hidden_layers_sizes[layer_index-1]
                connected_inputs_indices = np.random.choice(
                    range(len(self.hidden_layers[layer_index-1])),
                    size=np.random.randint(1, len(self.hidden_layers[layer_index-1]) + 1),
                    replace=False
                )
                self.hidden_hidden_connections[layer_index-1].append(connected_inputs_indices)
                new_neuron.w = np.random.normal(size=len(connected_inputs_indices))

            # Update connections to the next layer
            if layer_index < len(self.hidden_layers) - 1:
                # Ensure that hidden_hidden_connections[layer_index] exists
                if layer_index < len(self.hidden_hidden_connections):
                    for i in range(len(self.hidden_hidden_connections[layer_index])):
                        self.hidden_hidden_connections[layer_index][i] = np.append(self.hidden_hidden_connections[layer_index][i], len(layer) - 1)
            else:
                for i in range(len(self.hidden_output_connections)):
                    self.hidden_output_connections[i] = np.append(self.hidden_output_connections[i], len(layer) - 1)
        else:
            raise ValueError("Invalid layer index for adding neuron.")

    def remove_neuron(self, layer_index, neuron_id=None):
        """Удаляет нейрон из указанного скрытого слоя по ID."""
        if 0 <= layer_index < len(self.hidden_layers):
            layer = self.hidden_layers[layer_index]
            if neuron_id is None:
                raise ValueError("Neuron ID must be specified for removal.")

            # Находим индекс нейрона в списке скрытых нейронов
            try:
                neuron_index = next(i for i, neuron in enumerate(layer) if neuron.id == neuron_id)
            except StopIteration:
                raise ValueError(f"Neuron with ID {neuron_id} not found in layer {layer_index}.")

            # Remove neuron
            del layer[neuron_index]
            self.hidden_layers_sizes[layer_index] -= 1

            # Update connections
            if layer_index == 0 and neuron_index < len(self.input_hidden_connections):
                del self.input_hidden_connections[neuron_index]
            elif layer_index > 0 and layer_index - 1 < len(self.hidden_hidden_connections) and neuron_index < len(self.hidden_hidden_connections[layer_index-1]):
                del self.hidden_hidden_connections[layer_index-1][neuron_index]

            # Update connections from this layer to the next
            if layer_index < len(self.hidden_layers) - 1:
                for i in range(len(self.hidden_hidden_connections[layer_index])):
                    indices_to_remove = np.where(self.hidden_hidden_connections[layer_index][i] == neuron_index)[0]
                    self.hidden_hidden_connections[layer_index][i] = np.delete(self.hidden_hidden_connections[layer_index][i], indices_to_remove)
            else:
                for i in range(len(self.hidden_output_connections)):
                    indices_to_remove = np.where(self.hidden_output_connections[i] == neuron_index)[0]
                    self.hidden_output_connections[i] = np.delete(self.hidden_output_connections[i], indices_to_remove)

            # Update indices in connections after removal
            self.update_connections_after_removal(layer_index, neuron_index)

        else:
            raise ValueError("Invalid layer index for removing neuron.")

    def update_connections_after_removal(self, layer_index, removed_index):
        """Обновляет индексы связей после удаления нейрона."""
        # Update connections within the same layer
        if layer_index < len(self.hidden_layers) - 1:
            for i in range(len(self.hidden_hidden_connections[layer_index])):
                self.hidden_hidden_connections[layer_index][i][self.hidden_hidden_connections[layer_index][i] > removed_index] -= 1
        else:
            for i in range(len(self.hidden_output_connections)):
                self.hidden_output_connections[i][self.hidden_output_connections[i] > removed_index] -= 1

    def adapt_network_structure(self, inputs, targets):
        """Адаптирует структуру сети на основе активности нейронов и ошибки."""
        # Вычисление ошибки
        output_layer_outputs = self.predict(inputs)
        output_errors = np.array(targets) - np.array(output_layer_outputs)
        avg_error = np.mean(np.abs(output_errors))

        # Добавление нейрона
        if avg_error > self.neuron_addition_threshold:
            layer_index = np.random.randint(0, len(self.hidden_layers))
            self.add_neuron(layer_index)
            print(f"Добавлен новый нейрон в слой {layer_index} из-за высокой ошибки.")

        # Удаление нейрона
        for layer_index, layer in enumerate(self.hidden_layers):
            for i, neuron in enumerate(layer):
                if neuron.activity < self.neuron_removal_threshold:
                    self.remove_neuron(layer_index, neuron_id=neuron.id)
                    print(f"Удален нейрон {neuron.id} из слоя {layer_index} из-за низкой активности.")
                    break  # Чтобы избежать проблем с индексацией после удаления

        # Адаптация связей (пример: удаление слабых связей)
        for layer_index, layer in enumerate(self.hidden_layers):
            for i, neuron in enumerate(layer):
                if layer_index == 0:
                    connections = self.input_hidden_connections
                else:
                    connections = self.hidden_hidden_connections[layer_index-1]

                # Check if i is within the bounds of connections
                if i < len(connections):
                    connected_inputs_indices = connections[i]
                    # Check if connected_inputs_indices is not empty
                    if len(connected_inputs_indices) > 0:
                        # Ensure j is within the bounds of neuron.w
                        original_num_weights = len(neuron.w)
                        num_connections = len(connected_inputs_indices)
                        min_len = min(original_num_weights, num_connections)
                        # Ensure the size of neuron.w matches the number of connections
                        if len(neuron.w) != len(connected_inputs_indices):
                            if len(connected_inputs_indices) > len(neuron.w):
                                neuron.w = np.concatenate((neuron.w, np.random.normal(size=len(connected_inputs_indices) - len(neuron.w))))
                            else:
                                neuron.w = neuron.w[:len(connected_inputs_indices)]
                        indices_to_remove = []
                        for j in range(min_len):
                            if abs(neuron.w[j]) < self.connection_threshold:
                                indices_to_remove.append(j)

                        # Remove connections in reverse order to avoid index issues
                        for j in sorted(indices_to_remove, reverse=True):
                            if j < len(connected_inputs_indices) and j < len(neuron.w) and len(connected_inputs_indices) > 0:
                                # Ensure there are connections to remove
                                if len(connected_inputs_indices) > j:
                                    input_index = connected_inputs_indices[j]
                                    connections[i] = np.delete(connections[i], j)
                                    neuron.w = np.delete(neuron.w, j)
                                    print(f"Удалена слабая связь между скрытым нейроном {neuron.id} и входным нейроном {input_index}.")


    def visualize(self):
        """Визуализирует нейронную сеть с использованием Plotly."""
        G = nx.DiGraph()

        # Добавляем узлы (нейроны)
        for neuron in self.input_neurons:
            G.add_node(neuron.id, layer=0, type=neuron.type)

        # Add hidden layers
        layer_index = 1
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                G.add_node(neuron.id, layer=layer_index, type=neuron.type)
            layer_index += 1

        for neuron in self.output_neurons:
            G.add_node(neuron.id, layer=layer_index, type=neuron.type)

        # Add edges (connections)
        # Input to first hidden layer
        for i, hidden_neuron in enumerate(self.hidden_layers[0]):
            if i < len(self.input_hidden_connections):  # check if index is within bounds
                connected_inputs_indices = self.input_hidden_connections[i]
                for input_index in connected_inputs_indices:
                    G.add_edge(input_index, hidden_neuron.id)

        # Hidden layers to hidden layers
        for layer_index in range(len(self.hidden_layers) - 1):
            current_layer = self.hidden_layers[layer_index]
            next_layer = self.hidden_layers[layer_index + 1]
            connections = self.hidden_hidden_connections[layer_index]

            for i, next_neuron in enumerate(next_layer):
                 if i < len(connections): # check if index is within bounds
                    connected_inputs_indices = connections[i]
                    for input_index in connected_inputs_indices:
                        G.add_edge(current_layer[input_index].id, next_neuron.id)

        # Last hidden layer to output
        for i, output_neuron in enumerate(self.output_neurons):
             if i < len(self.hidden_output_connections): # check if index is within bounds
                connected_inputs_indices = self.hidden_output_connections[i]
                for input_index in connected_inputs_indices:
                    G.add_edge(self.hidden_layers[-1][input_index].id, output_neuron.id)

        # Определяем позиции узлов для визуализации слоев
        pos = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            layer = node_data.get("layer", 0)  # Default to input layer
            if layer == 0:  # Входной слой
                num_neurons_in_layer = len(self.input_neurons)
                position = (0, -2 * (node / num_neurons_in_layer - 0.5))  # Распределяем по вертикали
            elif 1 <= layer <= len(self.hidden_layers):  # Скрытые слои
                layer_index = layer - 1
                num_neurons_in_layer = self.hidden_layers_sizes[layer_index]
                position = (layer, -2 * ((node - len(self.input_neurons) - sum(self.hidden_layers_sizes[:layer_index])) / num_neurons_in_layer - 0.5))  # Распределяем по вертикали
            else:  # Выходной слой
                num_neurons_in_layer = len(self.output_neurons)
                position = (len(self.hidden_layers) + 1, -2 * ((node - len(self.input_neurons) - sum(self.hidden_layers_sizes)) / num_neurons_in_layer - 0.5))  # Распределяем по вертикали
            pos[node] = position

        # Создаем Plotly Scatter для узлов
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_types = nx.get_node_attributes(G, 'type')
        node_colors = ['blue' if node_types[node] == 'Input' else 'red' if node_types[node] == 'Hidden' else 'green' for node in G.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=20,
                line_width=2))
        node_text = [f"Id: {node}<br>Type: {node_types[node]}" for node in G.nodes()]
        node_trace.hovertext = node_text

        # Создаем Plotly Scatter для ребер
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()

# Пример использования
if __name__ == '__main__':
    # Создаем нейронную сеть с несколькими скрытыми слоями
    nn = NeuralNetwork(input_size=93, hidden_layers_sizes=[93, 98, 97], output_size=93, learning_rate=0.1,
                       neuron_addition_threshold=0.2, neuron_removal_threshold=0.1,
                       connection_threshold=0.1)

    # Генерируем случайные входные данные и целевые значения
    inputs = np.random.rand(93)
    targets = np.random.rand(93)

    # Обучаем сеть с динамической адаптацией структуры
    print("Начинаем обучение с динамической адаптацией структуры...\nСгенерированные целевые значения: ", targets)
    nn.train(inputs, targets, epochs=100)

    # Получаем предсказания сети после обучения
    predictions = nn.predict(inputs)
    print("\nИтог:", predictions)

    # Визуализируем сеть после обучения
    nn.visualize()

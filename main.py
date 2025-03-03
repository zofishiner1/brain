import numpy as np
import networkx as nx
import plotly.graph_objects as go

class Neuron:
    def __init__(self, number_of_weights=1, neuron_id=None, neuron_type="Input"):
        self.w = np.random.normal(size=number_of_weights)  # Случайные веса
        self.b = np.random.normal()  # Случайное смещение
        self.id = int(neuron_id)  # Преобразование идентификатора в int
        self.type = neuron_type  # Тип нейрона (входной или скрытый)

    def activate(self, inputs):
        x = np.dot(self.w, inputs) + self.b  # Взвешенная сумма
        return 1 / (1 + np.exp(-x))  # Сигмоидная функция активации

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_neurons = [Neuron(neuron_id=i, neuron_type="Input") for i in range(input_size)]
        self.hidden_neurons = [Neuron(neuron_id=i + input_size, neuron_type="Hidden") for i in range(hidden_size)]
        self.output_neurons = [Neuron(neuron_id=i + input_size + hidden_size, neuron_type="Output") for i in range(output_size)]

        self.input_hidden_connections = self.create_random_connections(input_size, hidden_size)
        self.hidden_output_connections = self.create_random_connections(hidden_size, output_size)

        # Инициализируем веса для скрытых и выходных нейронов, основываясь на соединениях
        for i, neuron in enumerate(self.hidden_neurons):
            neuron.w = np.random.normal(size=len(self.input_hidden_connections[i]))

        for i, neuron in enumerate(self.output_neurons):
            neuron.w = np.random.normal(size=len(self.hidden_output_connections[i]))


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

    def predict(self, inputs):
        # Проверяем, что размер входных данных соответствует количеству входных нейронов
        if len(inputs) != len(self.input_neurons):
            raise ValueError("Incorrect number of inputs provided.")

        # Выводим входные данные
        print(f"Input data: {inputs}")

        # Активируем скрытые нейроны на основе входных данных и связей
        hidden_layer_outputs = []
        for i, hidden_neuron in enumerate(self.hidden_neurons):
            connected_inputs_indices = self.input_hidden_connections[i]
            connected_inputs = [inputs[j] for j in connected_inputs_indices]
            hidden_output = hidden_neuron.activate(connected_inputs)
            hidden_layer_outputs.append(hidden_output)

            # Выводим промежуточные результаты для скрытого слоя
            print(f"Hidden Neuron {hidden_neuron.id}: Inputs: {connected_inputs}, Output: {hidden_output}")

        # Активируем выходные нейроны на основе выходов скрытого слоя и связей
        output_layer_outputs = []
        for i, output_neuron in enumerate(self.output_neurons):
            connected_hidden_indices = self.hidden_output_connections[i]
            connected_hidden_outputs = [hidden_layer_outputs[j] for j in connected_hidden_indices]
            output_output = output_neuron.activate(connected_hidden_outputs)
            output_layer_outputs.append(output_output)

            # Выводим промежуточные результаты для выходного слоя
            print(f"Output Neuron {output_neuron.id}: Inputs: {connected_hidden_outputs}, Output: {output_output}")

        return output_layer_outputs

    def visualize(self):
        """Визуализирует нейронную сеть с использованием Plotly."""
        G = nx.DiGraph()

        # Добавляем узлы (нейроны)
        for neuron in self.input_neurons:
            G.add_node(neuron.id, layer=0, type=neuron.type)
        for neuron in self.hidden_neurons:
            G.add_node(neuron.id, layer=1, type=neuron.type)
        for neuron in self.output_neurons:
            G.add_node(neuron.id, layer=2, type=neuron.type)


        # Добавляем ребра (связи) от входного к скрытому слою
        for i, hidden_neuron in enumerate(self.hidden_neurons):
            connected_inputs_indices = self.input_hidden_connections[i]
            for input_index in connected_inputs_indices:
                G.add_edge(input_index, hidden_neuron.id)

        # Добавляем ребра (связи) от скрытого к выходному слою
        for i, output_neuron in enumerate(self.output_neurons):
            connected_hidden_indices = self.hidden_output_connections[i]
            for hidden_index in connected_hidden_indices:
                G.add_edge(self.hidden_neurons[hidden_index].id, output_neuron.id)  # Используем id скрытых нейронов


        # Определяем позиции узлов для визуализации слоев
        pos = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            layer = node_data.get("layer", 0)  # Default to input layer

            if layer == 0:  # Входной слой
                num_neurons_in_layer = len(self.input_neurons)
                position = (0, -2 * (node / num_neurons_in_layer - 0.5))  # Распределяем по вертикали
            elif layer == 1:  # Скрытый слой
                num_neurons_in_layer = len(self.hidden_neurons)
                position = (1, -2 * ((node - len(self.input_neurons)) / num_neurons_in_layer - 0.5))  # Распределяем по вертикали
            else:  # Выходной слой
                num_neurons_in_layer = len(self.output_neurons)
                position = (2, -2 * ((node - len(self.input_neurons) - len(self.hidden_neurons)) / num_neurons_in_layer - 0.5))  # Распределяем по вертикали
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
    # Создаем нейронную сеть с 4 входными нейронами, 3 скрытыми и 2 выходными
    nn = NeuralNetwork(input_size=100, hidden_size=93, output_size=92)

    # Генерируем случайные входные данные
    inputs = np.random.rand(100)

    # Получаем предсказания сети и отображаем процесс вычислений
    predictions = nn.predict(inputs)
    print("\n\nPredictions:", predictions)

    # Визуализируем сеть
    nn.visualize()

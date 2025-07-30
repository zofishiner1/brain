import numpy as np
from PIL import Image
import networkx as nx
import plotly.graph_objects as go
import h5py
import os
import json
import logging
from datetime import datetime
import sys

class Neuron:
    def __init__(self, number_of_weights=1, neuron_id=None, neuron_type="Input", activation_function="sigmoid"):
        self.w = np.random.normal(size=number_of_weights)
        self.b = np.random.normal()
        self.id = int(neuron_id) if neuron_id is not None else None
        self.type = neuron_type
        self.activation_function = activation_function
        self.activity = 0

    def activate(self, inputs):
        # Преобразуем входные данные в numpy массив (если они еще не в таком формате)
        # Ожидаем, что inputs - это одномерный массив для одного примера или батча.
        # Если inputs - батч, то каждый нейрон должен обрабатывать только соответствующую часть.
        # В текущей реализации активация нейрона происходит для одного набора входов.
        inputs = np.array(inputs)
        
        # Убедимся, что inputs одномерный для dot-произведения с self.w
        if inputs.ndim > 1:
            # Если inputs - это батч, берем первый элемент или усредняем/суммируем.
            # В данном контексте, где Neuron.activate вызывается для одного экземпляра,
            # предполагается, что inputs уже подготовлен для конкретного нейрона.
            # Если это происходит в predict для батча, то логика активации должна быть изменена
            # или inputs должен быть уже подготовлен (например, inputs[sample_idx, :]).
            # В данном случае, я предполагаю, что inputs всегда будет вектором для данного нейрона.
            # Если inputs - это массив (N, D), где N - батч, D - признаки, то нужно решить,
            # как активировать один нейрон для всего батча.
            # В текущем коде predict подает inputs[0] для input_neurons, а дальше current_layer_outputs.
            # Поэтому inputs всегда должен быть 1D для каждого нейрона.
            pass

        # Приводим входные данные и веса к одинаковой форме
        # Эта логика подгонки весов и входов достаточно спорна, т.к. она динамически меняет веса.
        # Обычно веса имеют фиксированный размер в зависимости от числа входных связей.
        # Если размер входов не совпадает, это скорее ошибка в архитектуре/связях, а не повод для изменения весов.
        # Однако, если вы хотите сохранить такую адаптацию, то я оставил ее.
        # Важно убедиться, что inputs.ndim == 1, как предполагалось для np.dot.
        if inputs.shape[0] != self.w.shape[0]:
            # Динамическое изменение весов, если их размер не совпадает с количеством подключенных входов.
            # Это может быть источником непредсказуемого поведения.
            # Более надежный подход: при инициализации связей и весов убедиться, что они соответствуют.
            # Или добавить pad/truncation здесь, но это также может быть проблематично.
            # Для простоты и соответствия вашей задумке, я сохраняю эту логику.
            if inputs.shape[0] < self.w.shape[0]:
                self.w = self.w[:inputs.shape[0]] # Усекаем веса, если входов меньше
            else:
                inputs = inputs[:self.w.shape[0]] # Усекаем входы, если весов меньше

            # Если inputs.shape[0] все еще не совпадает после попытки подгонки, это ошибка.
            if inputs.shape[0] != self.w.shape[0]:
                raise ValueError(f"Input size {inputs.shape[0]} does not match weight size {self.w.shape[0]} after adjustment.")

        # Вычисление взвешенной суммы
        x = np.dot(self.w, inputs) + self.b
    
        # Применение активационной функции
        if self.activation_function == "sigmoid":
            output = 1 / (1 + np.exp(-x))
        elif self.activation_function == "relu":
            output = np.maximum(0, x)
        elif self.activation_function == "tanh":
            output = np.tanh(x)
        else:
            raise ValueError("Unknown activation function")
    
        self.activity = output
        return output

    def serialize(self):
        return {
            'w': self.w.tolist(),
            'b': self.b,
            'id': self.id,
            'type': self.type,
            'activation_function': self.activation_function,
            'activity': self.activity
        }

    @classmethod
    def deserialize(cls, data):
        neuron = cls(neuron_id=data['id'], neuron_type=data['type'], activation_function=data['activation_function'])
        neuron.w = np.array(data['w'])
        neuron.b = data['b']
        neuron.activity = data['activity']
        return neuron

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_sizes, output_size, learning_rate=0.01,
                 neuron_addition_threshold=0.9, neuron_removal_threshold=0.1,
                 dropout_rate=0.5, l1_lambda=0.01, l2_lambda=0.01,
                 layer_addition_threshold=0.95, layer_removal_threshold=0.05,
                 max_layers=5, adaptation_cooldown=10,
                 neuron_addition_counter_limit=5, neuron_removal_counter_limit=5,
                 layer_addition_counter_limit=10, layer_removal_counter_limit=10,
                 batch_size=32, patience=10, connection_threshold=0.2):
        
        self.setup_logging()
        
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes.copy()
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.patience = patience

        self.neuron_addition_threshold = neuron_addition_threshold
        self.neuron_removal_threshold = neuron_removal_threshold
        self.layer_addition_threshold = layer_addition_threshold
        self.layer_removal_threshold = layer_removal_threshold
        self.max_layers = max_layers
        self.connection_threshold = connection_threshold

        self.adaptation_cooldown = adaptation_cooldown
        self.orig_adapt = adaptation_cooldown
        self.neuron_addition_counter_limit = neuron_addition_counter_limit
        self.neuron_removal_counter_limit = neuron_removal_counter_limit
        self.layer_addition_counter_limit = layer_addition_counter_limit
        self.layer_removal_counter_limit = layer_removal_counter_limit

        self.input_neurons = [Neuron(neuron_id=i, neuron_type="Input") for i in range(input_size)]
        self.hidden_layers = []
        self.hidden_neurons = []
        neuron_id_counter = input_size

        for layer_idx, layer_size in enumerate(hidden_layers_sizes):
            # number_of_weights для новых нейронов должен быть размер предыдущего слоя
            prev_layer_size = input_size if layer_idx == 0 else hidden_layers_sizes[layer_idx-1]
            hidden_layer = [Neuron(number_of_weights=prev_layer_size, neuron_id=neuron_id_counter + i, neuron_type="Hidden") for i in range(layer_size)]
            self.hidden_layers.append(hidden_layer)
            self.hidden_neurons.extend(hidden_layer)
            neuron_id_counter += layer_size

        # number_of_weights для выходных нейронов должен быть размер последнего скрытого слоя или входного слоя
        prev_layer_size_output = hidden_layers_sizes[-1] if hidden_layers_sizes else input_size
        self.output_neurons = [Neuron(number_of_weights=prev_layer_size_output, neuron_id=neuron_id_counter + i, neuron_type="Output") for i in range(output_size)]

        self.input_hidden_connections = []
        self.hidden_hidden_connections = []
        self.hidden_output_connections = []

        # Инициализация связей
        self.initialize_connections()
        # Инициализация весов
        self.initialize_weights()

        self.layer_errors = []
        self.n = 5 # Не используется, можно удалить

        self.neuron_addition_counter = 0
        self.neuron_removal_counter = 0
        self.layer_addition_counter = 0
        self.layer_removal_counter = 0
        self.last_adaptation = 0
        self.best_loss = float('inf')
        self.counter = 0

    def setup_logging(self):
        """Настройка системы логирования"""
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_filename = os.path.join(log_dir, f"nn_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        self.logger = logging.getLogger('NeuralNetwork')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Удаляем существующие обработчики, чтобы избежать дублирования логов
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def create_random_connections(self, source_size, target_size):
        connections = []
        for _ in range(target_size):
            # Гарантируем, что выбирается хотя бы 1 источник, но не больше source_size
            num_connections = np.random.randint(1, source_size + 1)
            connected_sources = np.random.choice(
                range(source_size),
                size=num_connections,
                replace=False # Не допускаем повторений
            ).tolist() # Преобразуем в список для удобства сериализации
            connections.append(connected_sources)
        return connections

    def initialize_connections(self):
        if self.hidden_layers_sizes:
            self.input_hidden_connections = self.create_random_connections(self.input_size, self.hidden_layers_sizes[0])
            self.hidden_hidden_connections = []
            for i in range(len(self.hidden_layers_sizes) - 1):
                self.hidden_hidden_connections.append(self.create_random_connections(self.hidden_layers_sizes[i], self.hidden_layers_sizes[i+1]))
            self.hidden_output_connections = self.create_random_connections(self.hidden_layers_sizes[-1], self.output_size)
        else: # Нет скрытых слоев, напрямую к выходному слою
            self.hidden_output_connections = self.create_random_connections(self.input_size, self.output_size)

    def initialize_weights(self):
        # Инициализация весов для нейронов на основе их входных связей
        # Input-Hidden Layer
        if self.hidden_layers:
            for i, neuron in enumerate(self.hidden_layers[0]):
                # Проверяем, что connections[i] существует и не пуст
                if i < len(self.input_hidden_connections) and self.input_hidden_connections[i]:
                    neuron.w = np.random.normal(size=len(self.input_hidden_connections[i]))
                else:
                    # Если связей нет или ошибка в создании, инициализируем с 1 весом
                    neuron.w = np.random.normal(size=1) 
                    self.logger.warning(f"Нейрон {neuron.id} в первом скрытом слое не имеет входных связей. Инициализирован с одним весом.")

            # Hidden-Hidden Layers
            for layer_index in range(len(self.hidden_layers) - 1):
                for i, neuron in enumerate(self.hidden_layers[layer_index+1]):
                    if i < len(self.hidden_hidden_connections[layer_index]) and self.hidden_hidden_connections[layer_index][i]:
                        neuron.w = np.random.normal(size=len(self.hidden_hidden_connections[layer_index][i]))
                    else:
                        neuron.w = np.random.normal(size=1)
                        self.logger.warning(f"Нейрон {neuron.id} в скрытом слое {layer_index+1} не имеет входных связей. Инициализирован с одним весом.")
            
        # Hidden-Output Layer or Input-Output Layer (if no hidden layers)
        for i, neuron in enumerate(self.output_neurons):
            if i < len(self.hidden_output_connections) and self.hidden_output_connections[i]:
                neuron.w = np.random.normal(size=len(self.hidden_output_connections[i]))
            else:
                neuron.w = np.random.normal(size=1)
                self.logger.warning(f"Выходной нейрон {neuron.id} не имеет входных связей. Инициализирован с одним весом.")

    def predict(self, inputs, train_mode=True):
        # inputs ожидается как (num_samples, input_size)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1) # Преобразуем один пример в батч размером 1
        
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Incorrect number of inputs provided. Expected {self.input_size}, got {inputs.shape[1]}.")

        # Список для хранения выходов каждого слоя для каждого примера в батче
        # layer_outputs[0] = inputs (активности входного слоя)
        # layer_outputs[1] = активности первого скрытого слоя и т.д.
        # layer_outputs[-1] = активности последнего скрытого слоя
        # output_layer_outputs = активности выходного слоя
        batch_layer_outputs = [inputs] # Начальный слой - входные данные
        
        # Маски дропаута для каждого скрытого слоя в батче
        batch_dropout_masks = []

        current_inputs = inputs # Входы для текущего слоя, изначально это входные данные
        
        # Прямое распространение по скрытым слоям
        for layer_index, hidden_layer in enumerate(self.hidden_layers):
            next_layer_outputs = np.zeros((inputs.shape[0], len(hidden_layer))) # Выходы текущего скрытого слоя
            
            # Генерируем маску дропаута для текущего слоя
            # Маска применяется к выходам нейронов, а не к их активациям во время умножения на веса
            # Здесь маска генерируется для всего слоя, и если train_mode=True, то применяется
            # В противном случае, все нейроны остаются активными (маска из единиц)
            if train_mode:
                # np.random.binomial(n=1, p=1-self.dropout_rate, size=...) генерирует 0 или 1
                # size=len(hidden_layer) - маска для нейронов в слое
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=len(hidden_layer)) / (1 - self.dropout_rate)
            else:
                dropout_mask = np.ones(len(hidden_layer)) # В режиме тестирования dropout не применяется

            batch_dropout_masks.append(dropout_mask)

            # Определяем связи для текущего слоя
            if layer_index == 0:
                connections = self.input_hidden_connections
            else:
                connections = self.hidden_hidden_connections[layer_index - 1]

            # Вычисляем выход каждого нейрона в текущем скрытом слое
            for i, neuron in enumerate(hidden_layer):
                if i < len(connections): # Проверяем, что есть связи для этого нейрона
                    connected_inputs_indices = np.array(connections[i]) # Преобразуем в np.array для индексации
                    
                    # Проверяем, что connected_inputs_indices не пуст
                    if len(connected_inputs_indices) == 0:
                        self.logger.warning(f"Нейрон {neuron.id} в скрытом слое {layer_index} не имеет входных связей.")
                        # Если нет связей, активность нейрона может быть 0 или bias
                        next_layer_outputs[:, i] = neuron.activate(np.zeros(len(neuron.w))) * dropout_mask[i]
                        continue

                    # Проверка и подгонка размера весов нейрона к количеству его связей
                    # Это очень важный момент: если веса не соответствуют связям, то ошибка.
                    # Лучше на этапе add_neuron / initialize_weights гарантировать соответствие.
                    # Но если веса динамически меняются в activate(), то это должно быть консистентно.
                    # Здесь мы берем только те входы, которые соответствуют связям.
                    
                    # Получаем входы для данного нейрона из предыдущего слоя для всего батча
                    # current_inputs имеет форму (batch_size, prev_layer_size)
                    # connected_inputs_indices имеет форму (num_connections,)
                    
                    # Здесь происходит ключевое изменение:
                    # вместо `connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices]`
                    # мы должны выбрать столбцы из `current_inputs` для всего батча.
                    inputs_for_neuron = current_inputs[:, connected_inputs_indices]

                    # Если активационная функция принимает один набор входов для каждого вызова,
                    # то нужно делать это для каждого примера в батче.
                    # Это замедлит работу. Оптимально, если Neuron.activate может принимать батч.
                    # Пока Neuron.activate принимает один пример, придется итерировать по батчу.
                    # Это менее эффективно, чем матричные операции.
                    
                    # Исходя из Neuron.activate, которая ожидает 1D вход, итерируем по батчу.
                    # Это одно из узких мест для производительности.
                    neuron_output_for_batch = np.array([
                        neuron.activate(inputs_for_neuron[sample_idx, :]) for sample_idx in range(inputs.shape[0])
                    ])
                    
                    # Применяем дропаут. dropout_mask[i] - это скаляр для i-го нейрона.
                    next_layer_outputs[:, i] = neuron_output_for_batch * dropout_mask[i]

                else:
                    self.logger.warning(f"Нейрон {neuron.id} в скрытом слое {layer_index} не имеет связей в connections. Пропущен.")
                    # Может быть, стоит заполнить нулями или по-другому обработать.
                    # Например, активация только от bias
                    next_layer_outputs[:, i] = neuron.activate(np.zeros(len(neuron.w))) * dropout_mask[i]
            
            batch_layer_outputs.append(next_layer_outputs)
            current_inputs = next_layer_outputs # Выходы текущего слоя становятся входами для следующего
        
        # Вычисление выхода последнего слоя
        output_layer_outputs = np.zeros((inputs.shape[0], len(self.output_neurons)))
        
        # Входы для выходного слоя берутся из последнего скрытого слоя или входного, если скрытых нет
        if self.hidden_layers:
            prev_layer_outputs = batch_layer_outputs[-1] # Выходы последнего скрытого слоя (batch_size, last_hidden_size)
        else:
            prev_layer_outputs = batch_layer_outputs[0] # Входные данные (batch_size, input_size)

        connections = self.hidden_output_connections # Связи для выходного слоя

        for i, output_neuron in enumerate(self.output_neurons):
            if i < len(connections):
                connected_inputs_indices = np.array(connections[i])
                
                if len(connected_inputs_indices) == 0:
                    self.logger.warning(f"Выходной нейрон {output_neuron.id} не имеет входных связей.")
                    output_layer_outputs[:, i] = output_neuron.activate(np.zeros(len(output_neuron.w)))
                    continue

                # Выбираем столбцы из предыдущего слоя для всего батча
                inputs_for_neuron = prev_layer_outputs[:, connected_inputs_indices]

                # Активация для каждого примера в батче
                neuron_output_for_batch = np.array([
                    output_neuron.activate(inputs_for_neuron[sample_idx, :]) for sample_idx in range(inputs.shape[0])
                ])
                output_layer_outputs[:, i] = neuron_output_for_batch
            else:
                self.logger.warning(f"Выходной нейрон {output_neuron.id} не имеет связей в connections. Пропущен.")
                output_layer_outputs[:, i] = output_neuron.activate(np.zeros(len(output_neuron.w))) # Активация от bias

        # Возвращаем активности выходного слоя для батча, и маски дропаута для каждого скрытого слоя
        return output_layer_outputs, batch_dropout_masks, batch_layer_outputs # Добавил batch_layer_outputs для backprop

    def train(self, inputs, targets, epochs=100):
        # inputs: (num_samples, input_size)
        # targets: (num_samples, output_size) (должны быть one-hot закодированы)
        
        best_loss = float('inf')
        patience_counter = 0 # Изменил имя переменной для ясности
        early_stop = False
        self.logger.info(f"Начало обучения. Размер обучающей выборки: {len(inputs)}")

        num_samples = inputs.shape[0]

        for epoch in range(epochs):
            if early_stop:
                self.logger.info("Ранняя остановка активирована")
                break
            
            epoch_loss = 0
            correct_predictions = 0 # Используется для accuracy, работает для one-hot targets

            # Перемешиваем данные в каждой эпохе
            indices = np.random.permutation(num_samples)
            shuffled_inputs = inputs[indices]
            shuffled_targets = targets[indices]

            # Итерация по батчам
            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_samples)
                input_batch = shuffled_inputs[batch_start:batch_end]
                target_batch = shuffled_targets[batch_start:batch_end]

                # Проверяем cooldown перед адаптацией
                # Адаптация происходит после каждого батча
                if self.adaptation_cooldown <= 0:
                    # Передаем только первый пример из батча для адаптации, так как adapt_network_structure
                    # спроектирована для одного примера. Если нужна адаптация по батчу, ее нужно переделать.
                    self.adapt_network_structure(input_batch[0].reshape(1, -1), target_batch[0].reshape(1, -1), epoch=epoch)
                else:
                    self.adaptation_cooldown -= 1

                # Прямое распространение для текущего батча
                # output_layer_outputs: (batch_size, output_size)
                # dropout_masks: список массивов dropout_mask для каждого скрытого слоя, каждый (num_neurons_in_layer,)
                # batch_layer_outputs: список (batch_size, num_neurons_in_layer) для каждого слоя
                output_layer_outputs, dropout_masks, batch_layer_outputs = self.predict(input_batch, train_mode=True)
                
                # Вычисление ошибки для батча
                # target_batch: (batch_size, output_size)
                # output_layer_outputs: (batch_size, output_size)
                output_errors = target_batch - output_layer_outputs # (batch_size, output_size)

                # MSE Loss для батча
                loss = np.mean(output_errors ** 2)

                # Регуляризация (L1 и L2)
                l1_term = sum(np.sum(np.abs(neuron.w)) for neuron in self.hidden_neurons + self.output_neurons)
                l2_term = sum(np.sum(neuron.w ** 2) for neuron in self.hidden_neurons + self.output_neurons)
                total_loss = loss + self.l1_lambda * l1_term + self.l2_lambda * l2_term
                epoch_loss += total_loss * input_batch.shape[0] # Учитываем размер батча для суммирования по эпохе

                # Подсчет правильных предсказаний (для accuracy)
                # Предполагаем, что targets - one-hot
                predicted_classes = np.argmax(output_layer_outputs, axis=1) # (batch_size,)
                true_classes = np.argmax(target_batch, axis=1) # (batch_size,)
                correct_predictions += np.sum(predicted_classes == true_classes)
                
                # Обратное распространение ошибки по батчу
                # Обновление весов выходного слоя
                # output_errors: (batch_size, output_size)
                # output_layer_outputs: (batch_size, output_size)
                # batch_layer_outputs[-1]: (batch_size, last_hidden_size) - выходы последнего скрытого слоя
                
                # Дельта для выходного слоя: (batch_size, output_size)
                # dL/da = (y_true - y_pred) * f'(a)
                # f'(a) = output * (1 - output) для сигмоиды
                output_deltas = output_errors * self.sigmoid_derivative(output_layer_outputs)

                # Выходы предыдущего слоя для выходного слоя (т.е. последний скрытый слой)
                # Или входной слой, если скрытых нет
                if self.hidden_layers:
                    prev_layer_outputs_for_output_layer = batch_layer_outputs[-1] # (batch_size, last_hidden_size)
                else:
                    prev_layer_outputs_for_output_layer = batch_layer_outputs[0] # (batch_size, input_size)

                for i, output_neuron in enumerate(self.output_neurons):
                    connected_inputs_indices = np.array(self.hidden_output_connections[i])
                    
                    if len(connected_inputs_indices) == 0:
                        continue # Нет связей, нечего обновлять
                    
                    # Входы для этого нейрона из предыдущего слоя для всего батча: (batch_size, num_connected_inputs)
                    inputs_to_neuron_batch = prev_layer_outputs_for_output_layer[:, connected_inputs_indices]
                    
                    # Gradient for weights: dL/dw = dL/da * da/dw = delta * input
                    # delta_i: (batch_size,) для i-го выходного нейрона
                    # inputs_to_neuron_batch_j: (batch_size,) для j-го входа
                    # Обновляем каждый вес нейрона
                    
                    # Веса neuron.w: (num_connected_inputs,)
                    # input_to_neuron_batch: (batch_size, num_connected_inputs)
                    # output_deltas[:, i]: (batch_size,)
                    
                    # Домножаем (batch_size, 1) на (batch_size, num_connected_inputs) и усредняем
                    # (batch_size, num_connected_inputs) .T * (batch_size, 1) -> (num_connected_inputs, 1)
                    # np.dot(inputs_to_neuron_batch.T, output_deltas[:, i])
                    
                    # Градиент по весам (dL/dw)
                    # output_deltas[:, i] (batch_size, )
                    # inputs_to_neuron_batch (batch_size, num_connected_inputs)
                    weight_gradients = np.dot(inputs_to_neuron_batch.T, output_deltas[:, i]) / input_batch.shape[0]

                    # Градиент по смещению (dL/db)
                    bias_gradient = np.mean(output_deltas[:, i])

                    # Обновление весов
                    output_neuron.w += self.learning_rate * (
                        weight_gradients
                        - self.l1_lambda * np.sign(output_neuron.w) # Применяем к каждому весу
                        - self.l2_lambda * output_neuron.w # Применяем к каждому весу
                    )
                    output_neuron.b += self.learning_rate * bias_gradient
                
                # Обратное распространение ошибки по скрытым слоям
                # hidden_deltas будет хранить дельты для каждого скрытого слоя
                hidden_deltas = [np.zeros(layer_outputs.shape) for layer_outputs in batch_layer_outputs[1:]] # Исключаем входной слой

                # Идем в обратном порядке по скрытым слоям
                for layer_index in reversed(range(len(self.hidden_layers))):
                    current_hidden_layer_neurons = self.hidden_layers[layer_index]
                    
                    # Выходы текущего скрытого слоя (batch_size, current_layer_size)
                    current_layer_outputs = batch_layer_outputs[layer_index + 1] # +1 потому что batch_layer_outputs[0] это входы

                    if layer_index == len(self.hidden_layers) - 1: # Если это последний скрытый слой
                        next_layer_neurons = self.output_neurons
                        next_layer_connections = self.hidden_output_connections
                        # Дельты следующего слоя (batch_size, output_size)
                        next_layer_deltas = output_deltas 
                    else: # Если это промежуточный скрытый слой
                        next_layer_neurons = self.hidden_layers[layer_index + 1]
                        next_layer_connections = self.hidden_hidden_connections[layer_index]
                        # Дельты следующего скрытого слоя (batch_size, next_hidden_size)
                        next_layer_deltas = hidden_deltas[layer_index + 1] 

                    # Маска дропаута для этого слоя (num_neurons_in_layer,)
                    # Если train_mode=False, то dropout_mask будут все единицы.
                    current_dropout_mask = dropout_masks[layer_index] 
                    
                    # Вычисляем дельту для каждого нейрона в текущем скрытом слое
                    # hidden_deltas[layer_index]: (batch_size, current_layer_size)
                    
                    for i, neuron in enumerate(current_hidden_layer_neurons):
                        error_contribution_sum = np.zeros(input_batch.shape[0]) # (batch_size,)
                        # Проходим по нейронам следующего слоя, к которым текущий нейрон связан
                        for j, next_neuron in enumerate(next_layer_neurons):
                            # Проверяем, есть ли связь от текущего нейрона (i) к следующему (j)
                            # Ищем, присутствует ли индекс 'i' в connected_inputs_indices для next_neuron
                            if j < len(next_layer_connections):
                                next_neuron_connected_inputs = np.array(next_layer_connections[j])
                                # Находим индекс 'i' в списке connected_inputs_indices для next_neuron
                                # Если next_neuron_connected_inputs содержит 'i'
                                if i in next_neuron_connected_inputs:
                                    # Получаем индекс i-го нейрона в списке связей следующего нейрона
                                    input_idx_in_next_neuron_weights = np.where(next_neuron_connected_inputs == i)[0][0]
                                    
                                    # Вклад ошибки от следующего нейрона: delta_j * weight_ji
                                    # next_layer_deltas[:, j]: (batch_size,) - дельта j-го нейрона следующего слоя
                                    # next_neuron.w[input_idx_in_next_neuron_weights]: скалярный вес
                                    error_contribution_sum += next_layer_deltas[:, j] * next_neuron.w[input_idx_in_next_neuron_weights]
                        
                        # Дельта для текущего скрытого нейрона
                        hidden_deltas[layer_index][:, i] = error_contribution_sum * self.sigmoid_derivative(current_layer_outputs[:, i]) * current_dropout_mask[i]
                    
                    # Обновление весов текущего скрытого слоя
                    # current_hidden_layer_neurons - нейроны текущего скрытого слоя
                    # hidden_deltas[layer_index] - дельты для текущего скрытого слоя
                    # current_layer_outputs - выходы текущего скрытого слоя (используется для производной)
                    
                    # Входы для текущего скрытого слоя:
                    if layer_index == 0:
                        prev_layer_outputs_for_current_hidden_layer = batch_layer_outputs[0] # Входы сети
                        connections_for_current_layer = self.input_hidden_connections
                    else:
                        prev_layer_outputs_for_current_hidden_layer = batch_layer_outputs[layer_index] # Выходы предыдущего скрытого слоя
                        connections_for_current_layer = self.hidden_hidden_connections[layer_index - 1]

                    for i, neuron in enumerate(current_hidden_layer_neurons):
                        if i >= len(connections_for_current_layer):
                            continue # Если нет связей, пропускаем

                        connected_inputs_indices = np.array(connections_for_current_layer[i])
                        
                        if len(connected_inputs_indices) == 0:
                            continue # Нет связей, нечего обновлять
                        
                        # Входы для этого нейрона из предыдущего слоя для всего батча
                        inputs_to_neuron_batch = prev_layer_outputs_for_current_hidden_layer[:, connected_inputs_indices]
                        
                        # Градиент по весам
                        weight_gradients = np.dot(inputs_to_neuron_batch.T, hidden_deltas[layer_index][:, i]) / input_batch.shape[0]

                        # Градиент по смещению
                        bias_gradient = np.mean(hidden_deltas[layer_index][:, i])

                        # Обновление весов
                        neuron.w += self.learning_rate * (
                            weight_gradients
                            - self.l1_lambda * np.sign(neuron.w)
                            - self.l2_lambda * neuron.w
                        )
                        neuron.b += self.learning_rate * bias_gradient


            # Вычисляем среднюю ошибку и точность за эпоху
            avg_epoch_loss = epoch_loss / num_samples
            accuracy = correct_predictions / num_samples

            # Проверка на раннюю остановку
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0 # Сброс счетчика, если есть улучшение
                # Сохраняем лучшие веса
                self.best_weights = {
                    'hidden_neurons': [neuron.serialize() for layer in self.hidden_layers for neuron in layer],
                    'output_neurons': [neuron.serialize() for neuron in self.output_neurons]
                }
            else:
                patience_counter += 1
                # Если ошибка не улучшается, восстанавливаем лучшие веса, если patience_counter достиг порога
                # Восстановление при patience // 2 позволяет модели "откатиться" к лучшему состоянию
                # без полной остановки, давая шанс улучшиться позже.
                if patience_counter >= self.patience // 2 and hasattr(self, 'best_weights'):
                    self.logger.info(f"Восстановление лучших весов на эпохе {epoch+1}.")
                    # Здесь нужно аккуратно восстанавливать веса, так как структура могла измениться
                    # Восстанавливаем только те нейроны, которые существуют в текущей структуре
                    
                    # Восстанавливаем веса скрытых нейронов
                    for i, neuron_data in enumerate(self.best_weights['hidden_neurons']):
                        # Нужно найти соответствующий нейрон в self.hidden_neurons по ID,
                        # так как после адаптации порядок может измениться
                        found = False
                        for layer in self.hidden_layers:
                            for current_neuron in layer:
                                if current_neuron.id == neuron_data['id']:
                                    current_neuron.w = np.array(neuron_data['w'])
                                    current_neuron.b = neuron_data['b']
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            self.logger.warning(f"Нейрон {neuron_data['id']} из best_weights не найден в текущей структуре скрытых слоев.")
                    
                    # Восстанавливаем веса выходных нейронов
                    for i, neuron_data in enumerate(self.best_weights['output_neurons']):
                        found = False
                        for current_neuron in self.output_neurons:
                            if current_neuron.id == neuron_data['id']:
                                current_neuron.w = np.array(neuron_data['w'])
                                current_neuron.b = neuron_data['b']
                                found = True
                                break
                        if not found:
                            self.logger.warning(f"Нейрон {neuron_data['id']} из best_weights не найден в текущей структуре выходного слоя.")


            if patience_counter >= self.patience:
                stop_msg = f"Early stopping triggered at epoch {epoch+1}"
                print(stop_msg)
                self.logger.info(stop_msg)
                early_stop = True

            # Вывод статистики
            log_msg = (f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                      f"Accuracy: {accuracy:.2%}, Adapt cooldown: {self.adaptation_cooldown}, Patience: {patience_counter}")
            print(log_msg)
            self.logger.info(log_msg)

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def add_neuron(self, layer_index):
        if 0 <= layer_index < len(self.hidden_layers):
            layer = self.hidden_layers[layer_index]
            # ID нового нейрона должен быть уникальным
            all_neuron_ids = set([n.id for n in self.input_neurons] + 
                                 [n.id for l in self.hidden_layers for n in l] + 
                                 [n.id for n in self.output_neurons])
            new_neuron_id = max(all_neuron_ids) + 1 if all_neuron_ids else 0

            # Определяем размерность весов для нового нейрона
            if layer_index == 0:
                prev_layer_size = len(self.input_neurons)
            else:
                prev_layer_size = len(self.hidden_layers[layer_index - 1])

            new_neuron = Neuron(number_of_weights=prev_layer_size, neuron_id=new_neuron_id, neuron_type="Hidden")
            layer.append(new_neuron)
            self.hidden_neurons.append(new_neuron)
            self.hidden_layers_sizes[layer_index] += 1

            # Добавляем связи для нового нейрона
            # connected_sources - это индексы нейронов из предыдущего слоя
            connected_sources = np.random.choice(
                range(prev_layer_size),
                size=np.random.randint(1, prev_layer_size + 1),
                replace=False
            ).tolist()

            if layer_index == 0:
                self.input_hidden_connections.append(connected_sources)
            else:
                self.hidden_hidden_connections[layer_index - 1].append(connected_sources)
            
            # Инициализируем веса нового нейрона с учетом его связей
            new_neuron.w = np.random.normal(size=len(connected_sources))

            msg = f"Добавлен нейрон в слой {layer_index}, ID: {new_neuron.id}. Новый размер слоя: {len(layer)}"
            print(msg)
            self.logger.info(msg)
            self.update_connections_after_layer_change() # Пересчитать все связи для консистентности

        else:
            error_msg = f"Invalid layer index: {layer_index}. Cannot add neuron."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def remove_neuron(self, layer_index, neuron_id):
        if 0 <= layer_index < len(self.hidden_layers):
            layer = self.hidden_layers[layer_index]
            
            # Поиск нейрона по ID
            neuron_index = None
            for i, neuron in enumerate(layer):
                if neuron.id == neuron_id:
                    neuron_index = i
                    break

            if neuron_index is None:
                error_msg = f"Neuron with ID {neuron_id} not found in layer {layer_index}."
                self.logger.error(error_msg)
                return # Не вызываем исключение, чтобы не прерывать обучение

            # Удаление нейрона из слоя
            removed_neuron = layer.pop(neuron_index)
            self.hidden_layers_sizes[layer_index] -= 1
            # Обновляем self.hidden_neurons, так как в нем могут быть ссылки на удаленный нейрон
            self.hidden_neurons = [neuron for l in self.hidden_layers for neuron in l]

            # Удаление связей, ведущих К удаленному нейрону (это связи текущего слоя)
            if layer_index == 0:
                if neuron_index < len(self.input_hidden_connections):
                    del self.input_hidden_connections[neuron_index]
            else:
                if neuron_index < len(self.hidden_hidden_connections[layer_index - 1]):
                    del self.hidden_hidden_connections[layer_index - 1][neuron_index]

            # Теперь самое сложное: обновление связей, ведущих ОТ удаленного нейрона
            # Это влияет на следующий слой
            if layer_index < len(self.hidden_layers): # Если это не последний скрытый слой
                next_layer_connections = self.hidden_hidden_connections[layer_index]
                for connections_list_for_next_neuron in next_layer_connections:
                    # Удаляем все вхождения ID удаленного нейрона из списков связей следующего слоя
                    connections_list_for_next_neuron[:] = [conn for conn in connections_list_for_next_neuron if conn != removed_neuron.id]
            elif layer_index == len(self.hidden_layers) -1 : # Если это последний скрытый слой, то связи к выходному слою
                for connections_list_for_output_neuron in self.hidden_output_connections:
                    connections_list_for_output_neuron[:] = [conn for conn in connections_list_for_output_neuron if conn != removed_neuron.id]


            success_msg = f"Удален нейрон с ID {neuron_id} из слоя {layer_index}. Новый размер слоя: {len(layer)}"
            print(success_msg)
            self.logger.info(success_msg)
            
            # Обновляем все связи и веса после удаления, т.к. индексация могла измениться
            self.update_connections_after_layer_change()
        else:
            error_msg = f"Invalid layer index: {layer_index}. Must be between 0 and {len(self.hidden_layers)-1}."
            self.logger.error(error_msg)
            # raise ValueError(error_msg) # Не вызываем исключение, чтобы не прерывать обучение

    def add_layer(self, number_of_neurons, activation_function="sigmoid"):
        if len(self.hidden_layers) >= self.max_layers:
            self.logger.warning("Достигнуто максимальное количество скрытых слоев. Новый слой не будет добавлен.")
            return

        new_layer = []
        # ID новых нейронов должны быть уникальными
        all_neuron_ids = set([n.id for n in self.input_neurons] + 
                             [n.id for l in self.hidden_layers for n in l] + 
                             [n.id for n in self.output_neurons])
        current_max_id = max(all_neuron_ids) if all_neuron_ids else -1

        # Размер предыдущего слоя для инициализации весов новых нейронов
        if not self.hidden_layers: # Если добавляем первый скрытый слой
            prev_layer_size = self.input_size
        else: # Если добавляем слой после существующих скрытых слоев
            prev_layer_size = self.hidden_layers_sizes[-1]

        for i in range(number_of_neurons):
            new_neuron_id = current_max_id + 1 + i
            new_layer.append(Neuron(number_of_weights=prev_layer_size, neuron_id=new_neuron_id, activation_function=activation_function))
        
        self.hidden_layers.append(new_layer)
        self.hidden_neurons.extend(new_layer)
        self.hidden_layers_sizes.append(number_of_neurons) # Добавляем размер нового слоя

        # Обновляем все связи после добавления слоя
        self.update_connections_after_layer_change()

        msg = f"Добавлен слой с размером {number_of_neurons}. Общее количество скрытых слоев: {len(self.hidden_layers)}"
        print(msg)
        self.logger.info(msg)

    def remove_layer(self, layer_index):
        if 0 <= layer_index < len(self.hidden_layers):
            if len(self.hidden_layers) <= 1: # Ограничение: всегда должен быть хотя бы один скрытый слой (или обрабатывать случай его отсутствия)
                self.logger.warning("Невозможно удалить последний скрытый слой. Должен быть хотя бы один.")
                return

            removed_layer = self.hidden_layers.pop(layer_index)
            self.hidden_layers_sizes.pop(layer_index) # Корректное удаление размера слоя
            
            # Обновляем self.hidden_neurons, так как ссылки на нейроны удаленного слоя стали недействительны
            self.hidden_neurons = [neuron for layer in self.hidden_layers for neuron in layer]
            
            msg = f"Удален слой {layer_index}. Общее количество скрытых слоев: {len(self.hidden_layers)}"
            print(msg)
            self.logger.info(msg)
            
            # Обязательно пересчитываем связи и веса!
            self.update_connections_after_layer_change()
        else:
            error_msg = f"Invalid layer index: {layer_index}. Must be between 0 and {len(self.hidden_layers)-1}."
            self.logger.error(error_msg)
            # raise ValueError(error_msg) # Не вызываем исключение, чтобы не прерывать обучение

    # calculate_layer_error не используется
    # def calculate_layer_error(self, layer_output, target):
    #     return np.mean((np.array(target) - np.array(layer_output)) ** 2)

    def adapt_network_structure(self, inputs, targets, epoch):
        # inputs и targets здесь ожидаются как (1, N) и (1, M) соответственно,
        # так как train передает по одному сэмплу для адаптации.
        
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Неверное количество входов. Ожидалось {self.input_size}, получено {inputs.shape[1]}.")

        # predict здесь вызывается в train_mode=False, так как это не часть прямого распространения для обучения
        # Вычисляем выход и потерю для текущего сэмпла
        output_layer_outputs, _, _ = self.predict(inputs, train_mode=False) # Не нужны dropout_masks и batch_layer_outputs
        output_errors = targets - output_layer_outputs # (1, output_size)
        loss = np.mean(output_errors ** 2) # Скаляр

        structure_changed = False

        # Адаптация связей (удаление слабых связей)
        for layer_index, layer in enumerate(self.hidden_layers):
            connections_to_current_layer = None
            if layer_index == 0:
                connections_to_current_layer = self.input_hidden_connections
            else:
                connections_to_current_layer = self.hidden_hidden_connections[layer_index-1]

            # Итерируем по связям, ведущим к нейронам текущего слоя
            for i, neuron in enumerate(layer):
                if i < len(connections_to_current_layer):
                    connected_inputs_indices = connections_to_current_layer[i] # Это список!
                    
                    if not connected_inputs_indices: # Если связей нет, пропускаем
                        continue

                    # Проверяем и удаляем слабые связи
                    # Важно: neuron.w имеет длину, соответствующую len(connected_inputs_indices)
                    indices_to_remove_from_neuron = [] # Индексы в массиве весов нейрона
                    for j, input_node_idx in enumerate(connected_inputs_indices):
                        if j < len(neuron.w) and abs(neuron.w[j]) < self.connection_threshold:
                            indices_to_remove_from_neuron.append(j)

                    # Удаляем связи и соответствующие веса
                    for j in sorted(indices_to_remove_from_neuron, reverse=True):
                        # Удаляем из списка связей, ведущих к этому нейрону
                        removed_input_idx = connected_inputs_indices.pop(j)
                        # Удаляем соответствующий вес
                        neuron.w = np.delete(neuron.w, j)
                        msg = f"Удалена слабая связь между скрытым нейроном {neuron.id} (слой {layer_index}) и входным нейроном {removed_input_idx}."
                        print(msg)
                        self.logger.info(msg)
                        structure_changed = True
                    
                    # После удаления, если у нейрона не осталось весов, но связи есть, это проблема.
                    # Или если весов меньше, чем связей (из-за ошибки в логике), то тоже.
                    # Гарантируем, что число весов соответствует числу связей
                    if len(neuron.w) != len(connected_inputs_indices):
                         # Обычно это должна быть ошибка, но если допустимо динамическое изменение,
                         # то нужно добавить/удалить веса
                        if len(neuron.w) > len(connected_inputs_indices):
                            neuron.w = neuron.w[:len(connected_inputs_indices)]
                        else:
                            neuron.w = np.concatenate((neuron.w, np.random.normal(size=len(connected_inputs_indices) - len(neuron.w))))
        
        # Добавление нейрона
        if loss > self.neuron_addition_threshold:
            self.neuron_addition_counter += 1
            if self.neuron_addition_counter > self.neuron_addition_counter_limit:
                # Выбираем случайный скрытый слой, в который будем добавлять нейрон
                if self.hidden_layers:
                    layer_index_to_add = np.random.randint(0, len(self.hidden_layers))
                    self.add_neuron(layer_index_to_add)
                    self.neuron_addition_counter = 0
                    self.last_adaptation = epoch
                    structure_changed = True
                else:
                    self.logger.info("Нет скрытых слоев для добавления нейронов.")
        else:
            self.neuron_addition_counter = 0

        # Удаление нейрона
        if loss < self.neuron_removal_threshold:
            self.neuron_removal_counter += 1
            if self.neuron_removal_counter > self.neuron_removal_counter_limit:
                if self.hidden_layers:
                    # Выбираем случайный скрытый слой
                    layer_index_to_remove_from = np.random.randint(0, len(self.hidden_layers))
                    if self.hidden_layers[layer_index_to_remove_from]:
                        # Выбираем случайный нейрон для удаления
                        neuron_to_remove = np.random.choice(self.hidden_layers[layer_index_to_remove_from])
                        self.remove_neuron(layer_index_to_remove_from, neuron_to_remove.id)
                        self.neuron_removal_counter = 0
                        self.last_adaptation = epoch
                        structure_changed = True
                    else:
                        self.logger.info(f"Слой {layer_index_to_remove_from} пуст. Нечего удалять.")
                else:
                    self.logger.info("Нет скрытых слоев для удаления нейронов.")
        else:
            self.neuron_removal_counter = 0

        # Добавление слоя
        # Условие (loss > 1) является очень сильным. Если loss обычно меньше 1, слои никогда не добавятся.
        # Может быть, стоит использовать neuron_addition_threshold или другой порог.
        if len(self.hidden_layers) < self.max_layers and loss > self.layer_addition_threshold:
            self.layer_addition_counter += 1
            if self.layer_addition_counter > self.layer_addition_counter_limit:
                # Размер нового слоя можно брать как input_size или средний размер существующих слоев
                self.add_layer(number_of_neurons=self.input_size)
                self.layer_addition_counter = 0
                self.last_adaptation = epoch
                structure_changed = True
        else:
            self.layer_addition_counter = 0

        # Удаление слоя
        if len(self.hidden_layers) > 1 and loss < self.layer_removal_threshold:
            self.layer_removal_counter += 1
            if self.layer_removal_counter > self.layer_removal_counter_limit:
                # Выбираем случайный скрытый слой для удаления
                layer_index_to_remove = np.random.randint(0, len(self.hidden_layers))
                self.remove_layer(layer_index_to_remove)
                self.layer_removal_counter = 0
                self.last_adaptation = epoch
                structure_changed = True
        else:
            self.layer_removal_counter = 0

        # Сбрасываем cooldown только если была изменена структура сети
        if structure_changed:
            self.adaptation_cooldown = self.orig_adapt

    def visualize(self):
        """Визуализирует нейронную сеть с использованием Plotly."""
        G = nx.DiGraph()

        # Добавляем узлы (нейроны)
        node_id_to_layer_index = {} # Для определения слоя по ID нейрона
        node_id_to_local_index = {} # Для определения локального индекса в слое
        
        # Входной слой
        for i, neuron in enumerate(self.input_neurons):
            G.add_node(neuron.id, layer=0, type=neuron.type)
            node_id_to_layer_index[neuron.id] = 0
            node_id_to_local_index[neuron.id] = i

        # Скрытые слои
        layer_offset = len(self.input_neurons) # Смещение для ID нейронов в hidden_neurons
        for layer_idx, hidden_layer in enumerate(self.hidden_layers):
            for i, neuron in enumerate(hidden_layer):
                G.add_node(neuron.id, layer=layer_idx + 1, type=neuron.type)
                node_id_to_layer_index[neuron.id] = layer_idx + 1
                node_id_to_local_index[neuron.id] = i

        # Выходной слой
        output_layer_idx = len(self.hidden_layers) + 1
        for i, neuron in enumerate(self.output_neurons):
            G.add_node(neuron.id, layer=output_layer_idx, type=neuron.type)
            node_id_to_layer_index[neuron.id] = output_layer_idx
            node_id_to_local_index[neuron.id] = i


        # Добавляем ребра (связи)
        # Input to Hidden Layer 0
        if self.hidden_layers:
            for i, hidden_neuron in enumerate(self.hidden_layers[0]):
                if i < len(self.input_hidden_connections):
                    connected_inputs_indices = self.input_hidden_connections[i]
                    for input_idx in connected_inputs_indices:
                        # Убедимся, что input_idx - это ID входного нейрона, а не его локальный индекс.
                        # В create_random_connections используются range(source_size), что есть индексы.
                        # Значит, здесь это должны быть ID входных нейронов.
                        # В вашем коде input_neurons имеют ID от 0 до input_size-1.
                        # То есть input_idx - это сразу ID.
                        G.add_edge(self.input_neurons[input_idx].id, hidden_neuron.id)

            # Hidden to Hidden Layers
            for layer_index in range(len(self.hidden_layers) - 1):
                current_layer = self.hidden_layers[layer_index] # Текущий скрытый слой
                next_layer = self.hidden_layers[layer_index + 1] # Следующий скрытый слой
                connections = self.hidden_hidden_connections[layer_index] # Связи от current_layer к next_layer

                for i, next_neuron in enumerate(next_layer):
                    if i < len(connections):
                        connected_inputs_indices = connections[i] # Индексы нейронов из current_layer
                        for input_idx_in_prev_layer in connected_inputs_indices:
                            # input_idx_in_prev_layer - это локальный индекс нейрона в current_layer
                            # Нужно получить его глобальный ID
                            source_neuron_id = current_layer[input_idx_in_prev_layer].id
                            G.add_edge(source_neuron_id, next_neuron.id)

            # Last Hidden to Output Layer
            for i, output_neuron in enumerate(self.output_neurons):
                if i < len(self.hidden_output_connections):
                    connected_inputs_indices = self.hidden_output_connections[i] # Индексы нейронов из последнего скрытого слоя
                    for input_idx_in_prev_layer in connected_inputs_indices:
                        # input_idx_in_prev_layer - это локальный индекс нейрона в последнем скрытом слое
                        source_neuron_id = self.hidden_layers[-1][input_idx_in_prev_layer].id
                        G.add_edge(source_neuron_id, output_neuron.id)
        else: # No hidden layers, Input to Output Layer
            for i, output_neuron in enumerate(self.output_neurons):
                if i < len(self.hidden_output_connections):
                    connected_inputs_indices = self.hidden_output_connections[i] # Индексы нейронов из входного слоя
                    for input_idx_in_prev_layer in connected_inputs_indices:
                        source_neuron_id = self.input_neurons[input_idx_in_prev_layer].id
                        G.add_edge(source_neuron_id, output_neuron.id)


        # Определяем позиции узлов для визуализации слоев
        pos = {}
        # Используем node_id_to_layer_index и node_id_to_local_index для корректного позиционирования
        for node_id in G.nodes():
            layer = node_id_to_layer_index[node_id]
            local_idx = node_id_to_local_index[node_id]

            if layer == 0:  # Входной слой
                num_neurons_in_layer = len(self.input_neurons)
            elif layer <= len(self.hidden_layers):  # Скрытые слои
                num_neurons_in_layer = self.hidden_layers_sizes[layer - 1]
            else:  # Выходной слой
                num_neurons_in_layer = len(self.output_neurons)
            
            # Распределение по вертикали: чтобы нейроны располагались по центру, а не от верхнего края
            y_position = -2 * (local_idx - (num_neurons_in_layer - 1) / 2) / max(1, num_neurons_in_layer - 1) 
            pos[node_id] = (layer, y_position)

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
                # colorscale='YlGnBu', # Не использовать colorscale, если color задан напрямую
                reversescale=True,
                color=node_colors, # Используем заданные цвета
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
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='Layer'),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='Position in Layer'))
                        )
        fig.show()

    def update_connections_after_layer_change(self):
        # Эта функция должна быть вызвана после любого изменения структуры слоев/нейронов,
        # чтобы пересоздать все связи.

        # Важно: при удалении нейронов или слоев, старые ID нейронов,
        # которые были в связях, могут стать неактуальными.
        # create_random_connections создает новые связи, но не переводит старые ID.
        # Нужно быть уверенным, что после удаления нейронов, индексы, на которые ссылаются
        # связи, все еще действительны.
        # Простейший подход: полностью пересоздать все связи на основе текущих размеров слоев.

        self.initialize_connections() # Пересоздать все связи с нуля
        self.initialize_weights()     # Инициализировать веса для новых связей


    def save_to_file(self, filename):
        """Сохраняет параметры модели в файл."""
        model_params = {
            'input_size': self.input_size,
            'hidden_layers_sizes': self.hidden_layers_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'neuron_addition_threshold': self.neuron_addition_threshold,
            'neuron_removal_threshold': self.neuron_removal_threshold,
            'dropout_rate': self.dropout_rate,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'layer_addition_threshold': self.layer_addition_threshold,
            'layer_removal_threshold': self.layer_removal_threshold,
            'max_layers': self.max_layers,
            'adaptation_cooldown': self.adaptation_cooldown,
            'orig_adapt': self.orig_adapt,
            'neuron_addition_counter_limit': self.neuron_addition_counter_limit,
            'neuron_removal_counter_limit': self.neuron_removal_counter_limit,
            'layer_addition_counter_limit': self.layer_addition_counter_limit,
            'layer_removal_counter_limit': self.layer_removal_counter_limit,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'connection_threshold': self.connection_threshold,
            
            # Сохраняем нейроны полностью (их ID и параметры)
            'input_neurons': [neuron.serialize() for neuron in self.input_neurons],
            'hidden_neurons': [neuron.serialize() for layer in self.hidden_layers for neuron in layer],
            'output_neurons': [neuron.serialize() for neuron in self.output_neurons],
            
            # Связи хранятся как списки списков, потому что они являются list of lists of int
            # np.array(connections[i]) в predict требует, чтобы элементы были одномерными списками или массивами.
            'input_hidden_connections': self.input_hidden_connections,
            'hidden_hidden_connections': self.hidden_hidden_connections,
            'hidden_output_connections': self.hidden_output_connections,
        }
        try:
            with open(filename, 'w') as f:
                json.dump(model_params, f, indent=4) # indent для красивого форматирования JSON
            self.logger.info(f"Модель успешно сохранена в {filename}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели в {filename}: {e}")


    @classmethod
    def load_from_file(cls, filename):
        """Загружает параметры модели из файла."""
        try:
            with open(filename, 'r') as f:
                model_params = json.load(f)
        except FileNotFoundError:
            logging.error(f"Файл {filename} не найден.")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Ошибка декодирования JSON из файла {filename}: {e}")
            return None

        # Создаём экземпляр сети с параметрами из файла
        nn = cls(
            input_size=model_params['input_size'],
            hidden_layers_sizes=model_params['hidden_layers_sizes'],
            output_size=model_params['output_size'],
            learning_rate=model_params.get('learning_rate', 0.01), # .get() для обратной совместимости
            neuron_addition_threshold=model_params.get('neuron_addition_threshold', 0.9),
            neuron_removal_threshold=model_params.get('neuron_removal_threshold', 0.1),
            dropout_rate=model_params.get('dropout_rate', 0.5),
            l1_lambda=model_params.get('l1_lambda', 0.01),
            l2_lambda=model_params.get('l2_lambda', 0.01),
            layer_addition_threshold=model_params.get('layer_addition_threshold', 0.95),
            layer_removal_threshold=model_params.get('layer_removal_threshold', 0.05),
            max_layers=model_params.get('max_layers', 5),
            adaptation_cooldown=model_params.get('adaptation_cooldown', 10),
            neuron_addition_counter_limit=model_params.get('neuron_addition_counter_limit', 5),
            neuron_removal_counter_limit=model_params.get('neuron_removal_counter_limit', 5),
            layer_addition_counter_limit=model_params.get('layer_addition_counter_limit', 10),
            layer_removal_counter_limit=model_params.get('layer_removal_counter_limit', 10),
            batch_size=model_params.get('batch_size', 32),
            patience=model_params.get('patience', 10),
            connection_threshold=model_params.get('connection_threshold', 0.2)
        )

        # Восстанавливаем нейроны
        nn.input_neurons = [Neuron.deserialize(n) for n in model_params['input_neurons']]
        nn.hidden_neurons = [Neuron.deserialize(n) for n in model_params['hidden_neurons']]
        nn.output_neurons = [Neuron.deserialize(n) for n in model_params['output_neurons']]

        # Восстанавливаем структуру скрытых слоев
        nn.hidden_layers = []
        neuron_idx_in_list = 0
        for size in nn.hidden_layers_sizes:
            layer_neurons = []
            for i in range(size):
                if neuron_idx_in_list < len(nn.hidden_neurons):
                    layer_neurons.append(nn.hidden_neurons[neuron_idx_in_list])
                    neuron_idx_in_list += 1
                else:
                    logging.error("Несоответствие количества нейронов при загрузке скрытых слоев.")
                    break
            nn.hidden_layers.append(layer_neurons)
        
        # Восстанавливаем связи
        nn.input_hidden_connections = model_params['input_hidden_connections']
        nn.hidden_hidden_connections = model_params['hidden_hidden_connections']
        nn.hidden_output_connections = model_params['hidden_output_connections']
        
        # После загрузки необходимо убедиться, что все веса инициализированы корректно для связей
        # или переинициализировать их (хотя они уже загружены)
        # nn.initialize_weights() # Можно вызвать для проверки консистентности, но может перезаписать загруженные веса

        logging.info(f"Модель успешно загружена из {filename}")
        return nn

# Функция для предобработки одного изображения (ваша уже есть)
def preprocess_image(image_path, target_size=(28, 28)):
    if not os.path.exists(image_path):
        logging.error(f"Файл изображения не найден: {image_path}")
        return None
    try:
        image = Image.open(image_path).convert("L") # L - grayscale
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image) / 255.0
        image_vector = image_array.flatten()
        return image_vector
    except Exception as e:
        logging.error(f"Ошибка при предобработке изображения {image_path}: {e}")
        return None

def load_and_preprocess_data(folder_path, target_size=(28, 28), num_output_classes=33):
    # Русский алфавит А-Я
    russian_alphabet = [chr(code) for code in range(ord('А'), ord('Я') + 1)]
    # Метки от 0 до 32 для one-hot encoding
    label_map = {letter: idx for idx, letter in enumerate(russian_alphabet)}

    X, y = [], []

    if not os.path.exists(folder_path):
        logging.error(f"Папка с данными не найдена: {folder_path}")
        return np.array([]), np.array([])

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            # Извлекаем букву из имени файла (например, "А.png" -> "А")
            letter = os.path.splitext(filename)[0].upper()
            if letter in label_map:
                image_vector = preprocess_image(os.path.join(folder_path, filename), target_size)
                if image_vector is not None:
                    X.append(image_vector)
                    # Преобразуем метку в one-hot вектор
                    one_hot_label = np.zeros(num_output_classes)
                    one_hot_label[label_map[letter]] = 1
                    y.append(one_hot_label)
            else:
                msg = f"Пропущен файл {filename}: имя не в русском алфавите или не соответствует ожидаемому формату."
                logging.info(msg)
    
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # Настройка корневого логгера
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Убедимся, что basicConfig вызывается только один раз
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'application.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )

    # ---------------------------
    # Новый пример (выход 33 - классификация всех букв русского алфавита)
    # ---------------------------
    
    input_size_new = 28 * 28
    hidden_layers_sizes_new = [64]
    num_output_classes = 33  # 33 буквы русского алфавита

    logging.info("Инициализация новой нейронной сети для классификации русского алфавита.")
    nn_new = NeuralNetwork(
        input_size=input_size_new,
        hidden_layers_sizes=hidden_layers_sizes_new,
        output_size=num_output_classes, # Количество выходных классов
        learning_rate=0.01,
        neuron_addition_threshold=0.5, # Понизил для более частой адаптации
        neuron_removal_threshold=0.05, # Повысил для более частой адаптации
        dropout_rate=0.1, # Немного дропаута
        l1_lambda=0.001,
        l2_lambda=0.001,
        layer_addition_threshold=0.7, # Порог ошибки для добавления слоя
        layer_removal_threshold=0.02, # Порог ошибки для удаления слоя
        max_layers=5,
        adaptation_cooldown=5, # Уменьшил для более частой адаптации
        neuron_addition_counter_limit=5,
        neuron_removal_counter_limit=5,
        layer_addition_counter_limit=5,
        layer_removal_counter_limit=5,
        batch_size=32, # Уменьшил батч
        patience=10, # Увеличил терпение для ранней остановки
        connection_threshold=0.005 # Уменьшил порог для слабых связей
    )

    # Загрузка и предобработка данных
    # Убедитесь, что папка 'train_data' существует и содержит PNG изображения букв.
    data_folder = "train_data" 
    X_train, y_train = load_and_preprocess_data(data_folder, target_size=(28, 28), num_output_classes=num_output_classes)

    if X_train.shape[0] == 0:
        logging.error("Нет данных для обучения. Проверьте папку 'train_data' и названия файлов.")
    else:
        logging.info(f"Загружено {X_train.shape[0]} обучающих примеров.")
        nn_new.train(X_train, y_train, epochs=50) # Уменьшил эпохи для теста

        # Проверка предсказания на тестовом изображении
        test_image_path = "A_test.png" # Убедитесь, что этот файл существует
        test_input = preprocess_image(test_image_path)
        
        if test_input is not None:
            # predict возвращает (batch_size, output_size), поэтому берем [0]
            prediction, _, _ = nn_new.predict(test_input.reshape(1, -1), train_mode=False) 
            # Находим класс с максимальной вероятностью
            predicted_class_idx = np.argmax(prediction[0])
            russian_alphabet = [chr(code) for code in range(ord('А'), ord('Я') + 1)]
            predicted_letter = russian_alphabet[predicted_class_idx]
            print(f"Prediction for test input ({test_image_path}): {prediction[0]}, Predicted letter: {predicted_letter}")
            logging.info(f"Prediction for test input ({test_image_path}): {prediction[0]}, Predicted letter: {predicted_letter}")
        else:
            logging.error(f"Не удалось загрузить тестовое изображение: {test_image_path}")

        # Сохранение модели
        model_save_path = "nn_russian_alphabet.json"
        nn_new.save_to_file(model_save_path)

        # Создаём объект нейронной сети из файла
        logging.info(f"Загрузка модели из {model_save_path}")
        nn_loaded = NeuralNetwork.load_from_file(model_save_path)

        if nn_loaded:
            test_input_loaded = preprocess_image(test_image_path)
            if test_input_loaded is not None:
                prediction_loaded, _, _ = nn_loaded.predict(test_input_loaded.reshape(1, -1), train_mode=False)
                predicted_class_idx_loaded = np.argmax(prediction_loaded[0])
                predicted_letter_loaded = russian_alphabet[predicted_class_idx_loaded]
                print(f"Prediction for test input (loaded model) ({test_image_path}): {prediction_loaded[0]}, Predicted letter: {predicted_letter_loaded}")
                logging.info(f"Prediction for test input (loaded model) ({test_image_path}): {prediction_loaded[0]}, Predicted letter: {predicted_letter_loaded}")
            else:
                logging.error(f"Не удалось загрузить тестовое изображение для загруженной модели: {test_image_path}")

            # Визуализация загруженной модели
            logging.info("Визуализация загруженной модели.")
            nn_loaded.visualize()
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
        inputs = np.array(inputs)  # Преобразуем входные данные в numpy массив (если они еще не в таком формате)
    
        # Приводим входные данные и веса к одинаковой форме
        if inputs.ndim == 1 and self.w.ndim == 1 and inputs.shape[0] != self.w.shape[0]:
            if inputs.shape[0] < self.w.shape[0]:
                inputs = np.pad(inputs, (0, self.w.shape[0] - inputs.shape[0]), mode='constant')
            else:
                inputs = inputs[:self.w.shape[0]]
        elif inputs.ndim == 2 and self.w.ndim == 1 and inputs.shape[1] != self.w.shape[0]:
            inputs = inputs[:, :self.w.shape[0]]  # Подгонка размерности входа

        # Проверка размерностей
        if inputs.shape[0] != self.w.shape[0]:
            raise ValueError(f"Input size {inputs.shape[0]} does not match weight size {self.w.shape[0]}")

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
                 batch_size=32, patience=10, connection_threshold=0.2): # Добавлены batch_size и patience
        
        self.setup_logging()
        
        # Размеры слоев
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes.copy()
        self.output_size = output_size

        # Параметры обучения
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size  # Размер пакета
        self.patience = patience  # Параметр для ранней остановки

        # Пороги для добавления и удаления нейронов/слоев
        self.neuron_addition_threshold = neuron_addition_threshold
        self.neuron_removal_threshold = neuron_removal_threshold
        self.layer_addition_threshold = layer_addition_threshold
        self.layer_removal_threshold = layer_removal_threshold
        self.max_layers = max_layers
        self.connection_threshold = connection_threshold

        # Параметры для адаптации структуры сети
        self.adaptation_cooldown = adaptation_cooldown
        self.orig_adapt = adaptation_cooldown
        self.neuron_addition_counter_limit = neuron_addition_counter_limit
        self.neuron_removal_counter_limit = neuron_removal_counter_limit
        self.layer_addition_counter_limit = layer_addition_counter_limit
        self.layer_removal_counter_limit = layer_removal_counter_limit

        # Инициализация слоев
        self.input_neurons = [Neuron(neuron_id=i, neuron_type="Input") for i in range(input_size)]
        self.hidden_layers = []
        self.hidden_neurons = []
        neuron_id_counter = input_size

        for layer_size in hidden_layers_sizes:
            hidden_layer = [Neuron(number_of_weights=1, neuron_id=neuron_id_counter + i, neuron_type="Hidden") for i in range(layer_size)]
            self.hidden_layers.append(hidden_layer)
            self.hidden_neurons.extend(hidden_layer)
            neuron_id_counter += layer_size

        self.output_neurons = [Neuron(neuron_id=neuron_id_counter + i, neuron_type="Output") for i in range(output_size)]

        # Инициализация связей
        self.input_hidden_connections = self.create_random_connections(input_size, hidden_layers_sizes[0]) if hidden_layers_sizes else []
        self.hidden_hidden_connections = []
        for i in range(len(hidden_layers_sizes) - 1):
            self.hidden_hidden_connections.append(self.create_random_connections(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
        self.hidden_output_connections = self.create_random_connections(hidden_layers_sizes[-1], output_size) if hidden_layers_sizes else self.create_random_connections(input_size, output_size)

        # Инициализация весов
        self.initialize_weights()

        # Другие параметры
        self.layer_errors = []
        self.n = 5

        # Инициализация счетчиков для адаптации структуры
        self.neuron_addition_counter = 0
        self.neuron_removal_counter = 0
        self.layer_addition_counter = 0
        self.layer_removal_counter = 0
        self.last_adaptation = 0
        self.best_loss = float('inf')
        self.counter = 0

    def setup_logging(self):
        """Настройка системы логирования"""
        # Создаем папку logs, если ее нет
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Создаем имя файла с текущей датой и временем
        log_filename = os.path.join(log_dir, f"nn_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Настраиваем logging
        self.logger = logging.getLogger('NeuralNetwork')
        self.logger.setLevel(logging.INFO)
        
        # Форматтер для логов
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Обработчик для записи в файл
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Обработчик для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def create_random_connections(self, source_size, target_size):
        connections = []
        for _ in range(target_size):
            connected_sources = np.random.choice(
                range(source_size),
                size=np.random.randint(1, source_size + 1),
                replace=False
            )
            connections.append(connected_sources)
        return connections

    def initialize_weights(self):
        if self.hidden_layers:
            for i, neuron in enumerate(self.hidden_layers[0]):
                neuron.w = np.random.normal(size=len(self.input_hidden_connections[i]))
            for layer_index in range(len(self.hidden_layers) - 1):
                for i, neuron in enumerate(self.hidden_layers[layer_index+1]):
                    neuron.w = np.random.normal(size=len(self.hidden_hidden_connections[layer_index][i]))
            for i, neuron in enumerate(self.output_neurons):
                neuron.w = np.random.normal(size=len(self.hidden_output_connections[i]))
        else:
            self.hidden_output_connections = self.create_random_connections(self.input_size, self.output_size)
            for i, neuron in enumerate(self.output_neurons):
                neuron.w = np.random.normal(size=len(self.hidden_output_connections[i]))

    def predict(self, inputs):
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Incorrect number of inputs provided. Expected {self.input_size}, got {inputs.shape[1]}.")

        layer_outputs = [inputs[0]]
        dropout_masks = []

        for layer_index, hidden_layer in enumerate(self.hidden_layers):
            current_layer_outputs = []
            if layer_index == 0:
                connections = self.input_hidden_connections
                previous_layer_outputs = layer_outputs[-1]
            else:
                connections = self.hidden_hidden_connections[layer_index - 1]
                previous_layer_outputs = layer_outputs[-1]

            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=len(hidden_layer))
            dropout_masks.append(dropout_mask)

            for i, neuron in enumerate(hidden_layer):
                if len(connections) <= i:
                    continue

                connected_inputs_indices = connections[i]
                if len(connected_inputs_indices) != len(neuron.w):
                    num_missing = abs(len(connected_inputs_indices) - len(neuron.w))
                    if len(connected_inputs_indices) > len(neuron.w):
                        neuron.w = np.concatenate((neuron.w, np.random.normal(size=num_missing)))
                    else:
                        neuron.w = neuron.w[:len(connected_inputs_indices)]

                for j in connected_inputs_indices:
                    if j >= len(previous_layer_outputs):
                        raise IndexError(f"Index {j} out of range for previous_layer_outputs with length {len(previous_layer_outputs)}")
                    connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices if j < len(previous_layer_outputs)]
                    output = neuron.activate(connected_inputs) * dropout_mask[i]
                    current_layer_outputs.append(output)

            layer_outputs.append(current_layer_outputs)

        output_layer_outputs = []
        for i, output_neuron in enumerate(self.output_neurons):
            if self.hidden_layers:
                connections = self.hidden_output_connections
                previous_layer_outputs = layer_outputs[-1]
            else:
                connections = self.hidden_output_connections
                previous_layer_outputs = layer_outputs[0]

            connected_inputs_indices = connections[i]
            if len(connected_inputs_indices) != len(output_neuron.w):
                num_missing = abs(len(connected_inputs_indices) - len(output_neuron.w))
                if len(connected_inputs_indices) > len(output_neuron.w):
                    output_neuron.w = np.concatenate((output_neuron.w, np.random.normal(size=num_missing)))
                else:
                    output_neuron.w = output_neuron.w[:len(connected_inputs_indices)]

            connected_inputs = [previous_layer_outputs[j] for j in connected_inputs_indices if j < len(previous_layer_outputs)]
            output = output_neuron.activate(connected_inputs)
            output_layer_outputs.append(output)

        return output_layer_outputs, dropout_masks

    def train(self, inputs, targets, epochs=100):
        best_loss = float('inf')
        patience = self.patience if hasattr(self, 'patience') else 10
        counter = 0
        early_stop = False
        self.logger.info(f"Начало обучения. Размер обучающей выборки: {len(inputs)}")

        for epoch in range(epochs):
            if early_stop:
                self.logger.info("Ранняя остановка активирована")
                break
            
            epoch_loss = 0
            correct_predictions = 0

            # Перемешиваем данные в каждой эпохе
            indices = np.random.permutation(len(inputs))
            inputs = inputs[indices]
            targets = targets[indices]

            for idx in range(len(inputs)):
                input_sample = inputs[idx]
                target_sample = targets[idx]

                # Проверяем cooldown перед адаптацией
                if self.adaptation_cooldown <= 0:
                    self.adapt_network_structure(input_sample.reshape(1, -1), target_sample.reshape(1, -1), epoch=epoch)
                else:
                    self.adaptation_cooldown -= 1

                # Прямое распространение
                layer_outputs = [input_sample.tolist()]
                output_layer_outputs, dropout_masks = self.predict(input_sample.reshape(1, -1))
                layer_outputs += [mask.tolist() if isinstance(mask, np.ndarray) else mask for mask in dropout_masks]
                layer_outputs.append(output_layer_outputs)

                # Вычисление ошибки
                output_errors = target_sample - np.array(output_layer_outputs)

                # Регуляризация
                l1_term = sum(np.sum(np.abs(neuron.w)) for neuron in self.hidden_neurons + self.output_neurons)
                l2_term = sum(np.sum(neuron.w ** 2) for neuron in self.hidden_neurons + self.output_neurons)
                loss = np.mean(output_errors ** 2) + self.l1_lambda * l1_term + self.l2_lambda * l2_term
                epoch_loss += loss

                # Подсчет правильных предсказаний (для accuracy)
                if np.argmax(output_layer_outputs) == np.argmax(target_sample):
                    correct_predictions += 1

                # Обновление весов выходного слоя
                for i, output_neuron in enumerate(self.output_neurons):
                    connected_inputs_indices = self.hidden_output_connections[i]
                    for j, input_index in enumerate(connected_inputs_indices):
                        delta = output_errors[i] * self.sigmoid_derivative(output_layer_outputs[i])
                        output_neuron.w[j] += self.learning_rate * (
                            delta * layer_outputs[-2][int(input_index)]
                            - self.l1_lambda * np.sign(output_neuron.w[j])
                            - self.l2_lambda * output_neuron.w[j]
                        )
                    output_neuron.b += self.learning_rate * delta

                # Обратное распространение ошибки
                hidden_errors = [np.zeros(len(layer)) for layer in self.hidden_layers]

                for layer_index in reversed(range(len(self.hidden_layers))):
                    hidden_layer = self.hidden_layers[layer_index]
                    current_layer_outputs = layer_outputs[layer_index + 1]

                    if layer_index == len(self.hidden_layers) - 1:
                        next_layer_neurons = self.output_neurons
                        next_layer_connections = self.hidden_output_connections
                        next_layer_errors = output_errors
                        next_layer_outputs = output_layer_outputs
                    else:
                        next_layer_neurons = self.hidden_layers[layer_index + 1]
                        next_layer_connections = self.hidden_hidden_connections[layer_index]
                        next_layer_errors = hidden_errors[layer_index + 1]
                        next_layer_outputs = layer_outputs[layer_index + 2]

                    dropout_mask = dropout_masks[layer_index] if layer_index < len(dropout_masks) else np.ones(len(hidden_layer))

                    for i, neuron in enumerate(hidden_layer):
                        error_sum = 0
                        for j, next_neuron in enumerate(next_layer_neurons):
                            if j < len(next_layer_connections) and i in next_layer_connections[j]:
                                input_index_in_next_neuron = np.where(next_layer_connections[j] == i)[0][0]
                                delta = next_layer_errors[j] * self.sigmoid_derivative(next_layer_outputs[j])
                                error_sum += delta * next_neuron.w[input_index_in_next_neuron]
                        hidden_errors[layer_index][i] = error_sum * dropout_mask[i]

                    if layer_index == 0:
                        previous_layer_outputs = layer_outputs[0]
                        connections = self.input_hidden_connections
                    else:
                        previous_layer_outputs = layer_outputs[layer_index]
                        connections = self.hidden_hidden_connections[layer_index - 1]

                    for i, neuron in enumerate(hidden_layer):
                        if i < len(connections):
                            connected_inputs_indices = connections[i]
                            if len(connected_inputs_indices) > 0:
                                original_num_weights = len(neuron.w)
                                num_connections = len(connected_inputs_indices)
                                min_len = min(original_num_weights, num_connections)
                                if len(neuron.w) != len(connected_inputs_indices):
                                    if len(connected_inputs_indices) > len(neuron.w):
                                        neuron.w = np.concatenate(
                                            (neuron.w, np.random.normal(size=len(connected_inputs_indices) - len(neuron.w))))
                                    else:
                                        neuron.w = neuron.w[:len(connected_inputs_indices)]
                                for j in range(min_len):
                                    delta = hidden_errors[layer_index][i] * self.sigmoid_derivative(current_layer_outputs[i])
                                    neuron.w[j] += self.learning_rate * (
                                        delta * previous_layer_outputs[int(connected_inputs_indices[j])]
                                        - self.l1_lambda * np.sign(neuron.w[j])
                                        - self.l2_lambda * neuron.w[j]
                                    )
                                neuron.b += self.learning_rate * delta

            # Вычисляем среднюю ошибку и точность за эпоху
            avg_epoch_loss = epoch_loss / len(inputs)
            accuracy = correct_predictions / len(inputs)

            # Проверка на раннюю остановку
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                counter = 0
                # Сохраняем лучшие веса
                self.best_weights = {
                    'hidden_neurons': [neuron.serialize() for neuron in self.hidden_neurons],
                    'output_neurons': [neuron.serialize() for neuron in self.output_neurons]
                }
            else:
                counter += 1
                # Если ошибка не улучшается, восстанавливаем лучшие веса
                if counter >= patience // 2:
                    for i, neuron in enumerate(self.hidden_neurons):
                        neuron_data = self.best_weights['hidden_neurons'][i]
                        neuron.w = np.array(neuron_data['w'])
                        neuron.b = neuron_data['b']
                    for i, neuron in enumerate(self.output_neurons):
                        neuron_data = self.best_weights['output_neurons'][i]
                        neuron.w = np.array(neuron_data['w'])
                        neuron.b = neuron_data['b']

            if counter >= patience:
                stop_msg = f"Early stopping triggered at epoch {epoch+1}"
                print(stop_msg)
                self.logger.info(stop_msg)
                early_stop = True

            # Вывод статистики
            log_msg = (f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                      f"Accuracy: {accuracy:.2%}, Adapt cooldown: {self.adaptation_cooldown}")
            print(log_msg)
            self.logger.info(log_msg)


    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def add_neuron(self, layer_index):
        if 0 <= layer_index < len(self.hidden_layers):
            layer = self.hidden_layers[layer_index]
            new_neuron_id = max([neuron.id for neuron in self.hidden_neurons + self.input_neurons + self.output_neurons]) + 1
            new_neuron = Neuron(number_of_weights=1, neuron_id=new_neuron_id, neuron_type="Hidden")
            layer.append(new_neuron)
            self.hidden_neurons.append(new_neuron)
            self.hidden_layers_sizes[layer_index] += 1

            if layer_index == 0:
                connected_inputs_indices = np.random.choice(
                    range(len(self.input_neurons)),
                    size=np.random.randint(1, len(self.input_neurons) + 1),
                    replace=False
                )
                self.input_hidden_connections.append(connected_inputs_indices)
                new_neuron.w = np.random.normal(size=len(connected_inputs_indices))
                msg = f"Добавлен нейрон в слой {layer_index}, ID: {new_neuron.id}"
                print(msg)
                self.logger.info(msg)
            
            else:
                connected_inputs_indices = np.random.choice(
                    range(len(self.hidden_layers[layer_index - 1])),
                    size=np.random.randint(1, len(self.hidden_layers[layer_index - 1]) + 1),
                    replace=False
                )
                self.hidden_hidden_connections[layer_index - 1].append(connected_inputs_indices)
                new_neuron.w = np.random.normal(size=len(connected_inputs_indices))
                msg = f"Добавлен нейрон в слой {layer_index}, ID: {new_neuron.id}"
                print(msg)
                self.logger.info(msg)

    def remove_neuron(self, layer_index, neuron_id=None):
        if 0 <= layer_index < len(self.hidden_layers):
            layer = self.hidden_layers[layer_index]
            
            if neuron_id is None:
                error_msg = "Neuron ID must be specified for removal."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Поиск нейрона по ID
            neuron_index = None
            for i, neuron in enumerate(layer):
                if neuron.id == neuron_id:
                    neuron_index = i
                    break

            if neuron_index is None:
                error_msg = f"Neuron with ID {neuron_id} not found in layer {layer_index}."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Удаление нейрона
            del layer[neuron_index]
            self.hidden_layers_sizes[layer_index] -= 1
            self.hidden_neurons = [neuron for layer in self.hidden_layers for neuron in layer]

            # Удаление соответствующих связей
            if layer_index == 0:
                if neuron_index < len(self.input_hidden_connections):
                    del self.input_hidden_connections[neuron_index]
            else:
                if neuron_index < len(self.hidden_hidden_connections[layer_index - 1]):
                    del self.hidden_hidden_connections[layer_index - 1][neuron_index]

            success_msg = f"Удален нейрон с ID {neuron_id} из слоя {layer_index}. Новый размер слоя: {len(layer)}"
            print(success_msg)
            self.logger.info(success_msg)
            
            # Обновляем связи после удаления
            self.update_connections_after_layer_change()
        else:
            error_msg = f"Invalid layer index: {layer_index}. Must be between 0 and {len(self.hidden_layers)-1}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)


    def add_layer(self, number_of_neurons, activation_function="sigmoid"):
        new_layer = []
        new_neuron_id = max([neuron.id for neuron in self.hidden_neurons + self.input_neurons + self.output_neurons]) + 1
        for i in range(number_of_neurons):
            new_layer.append(Neuron(number_of_weights=self.input_size, neuron_id = new_neuron_id + i, activation_function=activation_function))
        self.hidden_layers.append(new_layer)
        self.hidden_neurons.extend(new_layer)
        self.hidden_layers_sizes.append(number_of_neurons)

        # Обновляем связи между слоями
        if len(self.hidden_layers) > 1:
            self.hidden_hidden_connections = []
            for i in range(len(self.hidden_layers) - 1):
                self.hidden_hidden_connections.append(self.create_random_connections(self.hidden_layers_sizes[i], self.hidden_layers_sizes[i+1]))

        # Обновляем связи между последним скрытым слоем и выходным слоем
        if self.hidden_layers:
            self.hidden_output_connections = self.create_random_connections(self.hidden_layers_sizes[-1], self.output_size)
        else:
            self.hidden_output_connections = self.create_random_connections(self.input_size, self.output_size)

        msg = f"Добавлен слой с размером {number_of_neurons}"
        print(msg)
        self.logger.info(msg)


    def remove_layer(self, layer_index):
        if 0 <= layer_index < len(self.hidden_layers):
            print(f"Удален слой {layer_index}")
            del self.hidden_layers[layer_index]
            del self.hidden_layers_sizes[layer_index]
            self.hidden_neurons = [neuron for layer in self.hidden_layers for neuron in layer]
            # После удаления обязательно пересчитываем связи и веса!
            self.update_connections_after_layer_change()
            msg = f"Удален слой {layer_index}"
            print(msg)
            self.logger.info(msg)


    def calculate_layer_error(self, layer_output, target):
        return np.mean((np.array(target) - np.array(layer_output)) ** 2)

    def adapt_network_structure(self, inputs, targets, epoch):
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Неверное количество входов. Ожидалось {self.input_size}, получено {inputs.shape[1]}.")

        output_layer_outputs, dropout_masks = self.predict(inputs)
        output_errors = targets - np.array(output_layer_outputs)
        loss = np.mean(output_errors ** 2)

        structure_changed = False

        # Адаптация связей (пример: удаление слабых связей)
        for layer_index, layer in enumerate(self.hidden_layers):
            for i, neuron in enumerate(layer):
                if layer_index == 0:
                    connections = self.input_hidden_connections
                else:
                    connections = self.hidden_hidden_connections[layer_index-1]

                if i < len(connections):
                    connected_inputs_indices = connections[i]
                    if len(connected_inputs_indices) > 0:
                        original_num_weights = len(neuron.w)
                        num_connections = len(connected_inputs_indices)
                        min_len = min(original_num_weights, num_connections)
                        if len(neuron.w) != len(connected_inputs_indices):
                            if len(connected_inputs_indices) > len(neuron.w):
                                neuron.w = np.concatenate((neuron.w, np.random.normal(size=len(connected_inputs_indices) - len(neuron.w))))
                            else:
                                neuron.w = neuron.w[:len(connected_inputs_indices)]
                        indices_to_remove = []
                        for j in range(min_len):
                            if abs(neuron.w[j]) < self.connection_threshold:
                                indices_to_remove.append(j)

                        for j in sorted(indices_to_remove, reverse=True):
                            if j < len(connected_inputs_indices) and j < len(neuron.w) and len(connected_inputs_indices) > 0:
                                if len(connected_inputs_indices) > j:
                                    input_index = connected_inputs_indices[j]
                                    connections[i] = np.delete(connections[i], j)
                                    neuron.w = np.delete(neuron.w, j)
                                    msg = f"Удалена слабая связь между скрытым нейроном {neuron.id} и входным нейроном {input_index}."
                                    print(msg)
                                    self.logger.info(msg)
                                    structure_changed = True

        # Добавление нейрона
        if loss > self.neuron_addition_threshold:
            self.neuron_addition_counter += 1
            if self.neuron_addition_counter > self.neuron_addition_counter_limit:
                layer_index = np.random.randint(0, len(self.hidden_layers))
                self.add_neuron(layer_index)
                self.neuron_addition_counter = 0
                self.last_adaptation = epoch
                structure_changed = True
        else:
            self.neuron_addition_counter = 0

        # Удаление нейрона
        if loss < self.neuron_removal_threshold:
            self.neuron_removal_counter += 1
            if self.neuron_removal_counter > self.neuron_removal_counter_limit:
                layer_index = np.random.randint(0, len(self.hidden_layers))
                if self.hidden_layers[layer_index]:
                    neuron_to_remove = np.random.choice(self.hidden_layers[layer_index])
                    self.remove_neuron(layer_index, neuron_to_remove.id)
                    self.neuron_removal_counter = 0
                    self.last_adaptation = epoch
                    structure_changed = True
        else:
            self.neuron_removal_counter = 0

        # Добавление слоя (только если loss > 1)
        if len(self.hidden_layers) < self.max_layers and loss > 1:
            self.layer_addition_counter += 1
            if self.layer_addition_counter > self.layer_addition_counter_limit:
                self.add_layer(number_of_neurons=self.input_size)
                self.hidden_layers_sizes.append(self.input_size)
                self.layer_addition_counter = 0
                self.last_adaptation = epoch
                structure_changed = True
        else:
            self.layer_addition_counter = 0

        # Удаление слоя
        if len(self.hidden_layers) > 1 and loss < self.layer_removal_threshold:
            self.layer_removal_counter += 1
            if self.layer_removal_counter > self.layer_removal_counter_limit:
                layer_index_to_remove = np.random.randint(0, len(self.hidden_layers))
                self.remove_layer(layer_index_to_remove)
                del self.hidden_layers_sizes[layer_index_to_remove]
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
        for neuron in self.input_neurons:
            G.add_node(neuron.id, layer=0, type=neuron.type)

        layer_index = 1
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                G.add_node(neuron.id, layer=layer_index, type=neuron.type)
            layer_index += 1

        for neuron in self.output_neurons:
            G.add_node(neuron.id, layer=layer_index, type=neuron.type)

        for i, hidden_neuron in enumerate(self.hidden_layers[0]):
            if i < len(self.input_hidden_connections):
                connected_inputs_indices = self.input_hidden_connections[i]
                for input_index in connected_inputs_indices:
                    G.add_edge(input_index, hidden_neuron.id)

        for layer_index in range(len(self.hidden_layers) - 1):
            current_layer = self.hidden_layers[layer_index]
            next_layer = self.hidden_layers[layer_index + 1]
            connections = self.hidden_hidden_connections[layer_index]

            for i, next_neuron in enumerate(next_layer):
                 if i < len(connections):
                    connected_inputs_indices = connections[i]
                    for input_index in connected_inputs_indices:
                        G.add_edge(current_layer[input_index].id, next_neuron.id)

        for i, output_neuron in enumerate(self.output_neurons):
             if i < len(self.hidden_output_connections):
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

    def update_connections_after_layer_change(self):
        # Пересчёт input_hidden_connections
        if self.hidden_layers:
            self.input_hidden_connections = self.create_random_connections(
                self.input_size, len(self.hidden_layers[0])
            )
        else:
            self.input_hidden_connections = []

        # Пересчёт hidden_hidden_connections
        self.hidden_hidden_connections = []
        for i in range(len(self.hidden_layers) - 1):
            self.hidden_hidden_connections.append(
                self.create_random_connections(len(self.hidden_layers[i]), len(self.hidden_layers[i+1]))
            )

        # Пересчёт hidden_output_connections
        if self.hidden_layers:
            self.hidden_output_connections = self.create_random_connections(
                len(self.hidden_layers[-1]), self.output_size
            )
        else:
            self.hidden_output_connections = self.create_random_connections(
                self.input_size, self.output_size
            )

        # Пересчёт весов нейронов
        self.initialize_weights()


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
            # Сохраняем нейроны полностью
            'hidden_neurons': [neuron.serialize() for neuron in self.hidden_neurons],
            'output_neurons': [neuron.serialize() for neuron in self.output_neurons],
            # Сохраняем структуру связей
            'input_hidden_connections': [conn.tolist() for conn in self.input_hidden_connections],
            'hidden_hidden_connections': [[conn.tolist() for conn in layer] for layer in self.hidden_hidden_connections],
            'hidden_output_connections': [conn.tolist() for conn in self.hidden_output_connections],
        }
        with open(filename, 'w') as f:
            json.dump(model_params, f)


    @classmethod
    def load_from_file(cls, filename):
        """Загружает параметры модели из файла."""
        with open(filename, 'r') as f:
            model_params = json.load(f)

        # Создаём экземпляр сети с нужными размерами
        nn = cls(
            input_size=model_params['input_size'],
            hidden_layers_sizes=model_params['hidden_layers_sizes'],
            output_size=model_params['output_size'],
            learning_rate=model_params['learning_rate'],
            neuron_addition_threshold=model_params['neuron_addition_threshold'],
            neuron_removal_threshold=model_params['neuron_removal_threshold'],
            dropout_rate=model_params['dropout_rate'],
            l1_lambda=model_params['l1_lambda'],
            l2_lambda=model_params['l2_lambda'],
            layer_addition_threshold=model_params['layer_addition_threshold'],
            layer_removal_threshold=model_params['layer_removal_threshold'],
            max_layers=model_params['max_layers']
        )

        # Восстанавливаем нейронов
        nn.hidden_neurons = [Neuron.deserialize(n) for n in model_params['hidden_neurons']]
        nn.output_neurons = [Neuron.deserialize(n) for n in model_params['output_neurons']]

        # Восстанавливаем структуру слоев
        nn.hidden_layers = []
        idx = 0
        for size in nn.hidden_layers_sizes:
            nn.hidden_layers.append(nn.hidden_neurons[idx:idx+size])
            idx += size

        # Восстанавливаем связи
        nn.input_hidden_connections = [np.array(conn) for conn in model_params['input_hidden_connections']]
        nn.hidden_hidden_connections = [[np.array(conn) for conn in layer] for layer in model_params['hidden_hidden_connections']]
        nn.hidden_output_connections = [np.array(conn) for conn in model_params['hidden_output_connections']]

        return nn


# Функция для предобработки одного изображения (ваша уже есть)
def preprocess_image(image_path, target_size=(28, 28)):
    image = Image.open(image_path).convert("L")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0
    image_vector = image_array.flatten()
    return image_vector

def load_and_preprocess_data(folder_path, target_size=(28, 28)):
    # Русский алфавит А-Я
    russian_alphabet = [chr(code) for code in range(ord('А'), ord('Я') + 1)]
    label_map = {letter: idx + 1 for idx, letter in enumerate(russian_alphabet)}  # метки от 1 до 33

    X, y = [], []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            letter = os.path.splitext(filename)[0].upper()
            if letter in label_map:
                # Загрузка и предобработка изображения
                image = Image.open(os.path.join(folder_path, filename)).convert("L")
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                image_array = np.array(image) / 255.0
                image_vector = image_array.flatten()

                X.append(image_vector)
                y.append(label_map[letter])
            else:
                msg = f"Пропущен файл {filename}: имя не в русском алфавите"
                print(msg)
                logging.info(msg)
    
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # Настройка корневого логгера
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'application.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # ---------------------------
    # Старый пример (выход 1 - бинарная классификация буквы "А")
    # ---------------------------
    """
    input_size_old = 28 * 28
    hidden_layers_sizes_old = [64]
    output_size_old = 1  # бинарная классификация

    nn = NeuralNetwork(
        input_size=input_size_old,
        hidden_layers_sizes=hidden_layers_sizes_old,
        output_size=output_size_old,
        learning_rate=0.01,
        neuron_addition_threshold=0.01,
        neuron_removal_threshold=0.11,
        dropout_rate=0.0,
        l1_lambda=0.0,
        l2_lambda=0.0,
        layer_addition_threshold=0.01,
        layer_removal_threshold=0.01,
        max_layers=10,
        adaptation_cooldown=0,
        neuron_addition_counter_limit=10,
        neuron_removal_counter_limit=10,
        layer_addition_counter_limit=10,
        layer_removal_counter_limit=10,
        batch_size=64,
        patience=10,
        connection_threshold=0.01
    )

    # Старый пример - раскомментирован
    image_path = "A.png"  # путь к изображению с буквой "А"
    input_data = preprocess_image(image_path)

    target_data = np.array([1])  # метка для буквы "А"

    nn.train(np.array([input_data]), target_data, epochs=100)

    # Сохранение модели
    nn.save_to_file("nn.json")
    
    test_image_path = "A.png"
    test_input = preprocess_image(test_image_path).reshape(1, -1)
    prediction = nn.predict(test_input)
    print("Prediction for test input:", prediction)

    # Создаём объект нейронной сети
    nn_test = NeuralNetwork.load_from_file("nn.json")

    test_image_path = "A_test.png"
    test_input = preprocess_image(test_image_path).reshape(1, -1)
    prediction = nn_test.predict(test_input)
    print("Prediction for test input(loaded):", prediction)
    
    nn_test.visualize()
    """
    # ---------------------------
    # Новый пример (выход 33 - классификация всех букв русского алфавита)
    # ---------------------------
    
    input_size_new = 28 * 28
    hidden_layers_sizes_new = [64]
    output_size_new = 33  # 33 буквы русского алфавита

    nn_new = NeuralNetwork(
        input_size=input_size_new,
        hidden_layers_sizes=hidden_layers_sizes_new,
        output_size=output_size_new,
        learning_rate=0.01,
        neuron_addition_threshold=0.01,
        neuron_removal_threshold=0.11,
        dropout_rate=0.0,
        l1_lambda=0.0,
        l2_lambda=0.0,
        layer_addition_threshold=0.01,
        layer_removal_threshold=0.01,
        max_layers=10,
        adaptation_cooldown=20,
        neuron_addition_counter_limit=10,
        neuron_removal_counter_limit=10,
        layer_addition_counter_limit=10,
        layer_removal_counter_limit=10,
        batch_size=64,
        patience=1,
        connection_threshold=0.01
    )

    X_train, y_train = load_and_preprocess_data("train_data", target_size=(28, 28))

    nn_new.train(X_train, y_train, epochs=100)

    test_image_path = "А_test.png"
    test_input = preprocess_image(test_image_path).reshape(1, -1)
    prediction = nn_new.predict(test_input)
    print("Prediction for test input (new model):", prediction)

    # Сохранение модели
    nn_new.save_to_file("nn.json")

    # Создаём объект нейронной сети
    nn_test = NeuralNetwork.load_from_file("nn.json")

    test_input_loaded = preprocess_image(test_image_path).reshape(1, -1)
    prediction_loaded = nn_test.predict(test_input_loaded)
    print("Prediction for test input (loaded model):", prediction_loaded)

    nn_test.visualize()

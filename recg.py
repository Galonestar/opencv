#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/21 下午5:29
# @Author : Galonestar
# @Version : V 1.0
# @File : recg.py
# @desc :

import cv2
import collections
import random
import numpy as np
import numpy as np
import joblib




parameters = {
    "epochs": 10000,
    "eta": 0.5,
    "mini_batch_size": 2,
    "threshold": 0.001
}

# 处理图片
def get_color(frame):
    print('go in get_color')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d

    return color


def getColorList():
    dict = collections.defaultdict(list)
    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list
    # 灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray'] = color_list
    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list
    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list
    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list
    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list
    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict


# 待识别图片
img = cv2.imread('img.jpg', 1)

# 区域，可改为for循环，同时识别多个区域
x2, y2, w2, h2 = (426, 909, 37, 37)
# print(x2, y2, w2, h2)
img_roi2 = img[int(y2):int(y2 + h2), int(x2):int(x2 + w2)]
cv2.rectangle(img=img, pt1=(x2, y2), pt2=(x2 + w2, y2 + h2), color=(0, 0, 255), thickness=1)
color = get_color(img_roi2)
cv2.putText(img, '{0}'.format(color),
            (x2 + 50, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 0, 255), 4,
            cv2.LINE_AA)

cv2.imshow('roi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


class NeuralNetwork(object):
    def __init__(self, layer_dimensions):
        # the number of the layers in NeuralNetwork
        self.no_of_layers = len(layer_dimensions)
        self.layer_dimensions = layer_dimensions
        # Custom weight initialization (course 4, Neural Networks) with mu=0 and std = 1/sqrt(number of connections for that neuron)
        self.biases = [np.random.randn(y, 1) for y in layer_dimensions[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(layer_dimensions[:-1], layer_dimensions[1:])]

    def feedforward(self, a):
        """This function is only used to get activations for evaluation"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def fit(self, training_data, epochs, eta=0.5, mini_batch_size=1, threshold=0.001):
        no_of_mini_batches = len(training_data)//mini_batch_size
        if len(training_data) % mini_batch_size > 0:
            no_of_mini_batches += 1

        for j in range(epochs):
            # shuffle
            random.shuffle(training_data)
            for i in range(0, no_of_mini_batches):
                mini_batch = training_data[i *
                                           mini_batch_size: (i + 1)*mini_batch_size]
                self.update_mini_batch(mini_batch, eta, len(training_data))

            # testing accuracy
            nailed_cases = self.get_nailed_cases(training_data)
            cost = self.calculate_cost(training_data)
            print(
                f"Epoch {j}: {nailed_cases}/{len(training_data)}, cost: {cost}\n")
            if cost <= threshold:
                break

    def get_predictions(self, X):
        return [np.argmax(self.feedforward(x)) for x in X]

    def update_mini_batch(self, mini_batch, eta, training_data_length):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases
        self.weights = [w + vw for w, vw in zip(self.weights, nabla_w)]
        self.biases = [b + vb for b, vb in zip(self.biases, nabla_b)]

    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (y - activations[-1]) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.no_of_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def get_nailed_cases(self, test_data):
        """Returns how many cases it nailed from the test_data"""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def calculate_cost(self, training_data):
        truth = [x[1] for x in training_data]
        outputs = [self.feedforward(x[0]) for x in training_data]
        suma = 0
        for y, t in zip(outputs, truth):
            cost_for_this_instance = (y - t)**2
            suma += sum(cost_for_this_instance)

        return suma/(2 * len(outputs))


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def save_results(model, model_name):
    print('Serializing model...')
    with open(model_name, 'wb') as f:
        joblib.dump(model, f)


def get_saved_model(model_name='model.pkl'):
    print('Deserializing model...')
    with open(model_name, 'rb') as f:
        model = joblib.load(f)
    return model


def get_option():
    option = int(input('Enter 1 for training, 2 for prediction: '))
    if option == 1:
        return 'train'
    elif option == 2:
        return 'predict'
    return None


def get_train_data():
    train_data = []
    with open('segments.data', 'r') as f:
        for index, line in enumerate(f):
            if index is 0:
                continue
            values = line.split(',')
            values = [int(x) for x in values]
            x = np.array(values[:7]).reshape((7, 1))
            y = np.array(values[7:]).reshape((10, 1))
            train_data.append((x, y))
    return train_data


def train_network():
    print('Getting train data...')
    train_data = get_train_data()
    model = NeuralNetwork((7, 10, 10))
    print('Training network...')
    model.fit(training_data=train_data, **parameters)
    save_results(model, "model.pkl")


def predict_led():
    model = get_saved_model()
    x = input('Enter the digit values separated by space: ')
    x = [int(val) for val in x.split(' ')]
    x = np.array(x).reshape((7, 1))
    predictions = model.get_predictions([x])
    print(f'I think this is a {predictions[0]}')


def main():
    option = get_option()
    if option == 'train':
        train_network()
    elif option == 'predict':
        predict_led()


if __name__ == '__main__':
    main()
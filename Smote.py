##############过采样
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

samples_per_class = {0: 5000, 1: 200, 2: 500, 3: 1500, 4: 350, 5: 800, 6: 200, 7: 400, 8: 1000, 9: 3000}
indices_to_keep = []

for class_index, num_samples in samples_per_class.items():
    indices = np.where(y_train == class_index)[0]

    if len(indices) < num_samples:
        # 如果少数类样本数量不足，进行过采样（随机复制）
        selected_indices = np.random.choice(indices, size=num_samples, replace=True)
    else:
        # 如果已经有足够的样本，随机选择不重复样本
        selected_indices = np.random.choice(indices, size=num_samples, replace=False)

    indices_to_keep.extend(selected_indices)

x_train_oversampled = x_train[indices_to_keep]
y_train_oversampled = y_train[indices_to_keep]

x_train_oversampled = x_train_oversampled.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_oversampled = to_categorical(y_train_oversampled, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print("过采样后的训练集形状:", x_train_oversampled.shape, y_train_oversampled.shape)


def cifar10_deeper_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(96, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


def get_true_labels(y_array):
    true_labels = np.argmax(y_array, axis=1)
    return true_labels


def p_mean(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    precisions = np.diag(cm) / np.sum(cm, axis=0)
    p_means = []

    for i in range(num_classes):
        other_classes = [j for j in range(num_classes) if j != i]
        other_recall = np.diag(cm)[other_classes] / np.sum(cm[other_classes], axis=1)
        p_mean = np.exp(np.mean(np.log(other_recall)))
        p_means.append(p_mean)

    return p_means


def g_mean(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    specificities = []

    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - np.diag(cm)[i])
        fp = np.sum(cm[:, i]) - np.diag(cm)[i]
        fn = np.sum(cm[i, :]) - np.diag(cm)[i]
        tp = np.diag(cm)[i]
        specificity = tn / (tn + fp)
        g_mean = np.sqrt(sensitivities[i] * specificity)
        specificities.append(g_mean)

    return specificities


def train_CNN(x_train, y_train, x_test, y_test, epochs=None, batch_size=None, seed=50):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = cifar10_deeper_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    predictions = model.predict(x_test)

    binary_predictions = np.argmax(predictions, axis=1)

    true_labels = get_true_labels(y_test)

    f1 = f1_score(true_labels, binary_predictions, average='weighted')
    F1 = classification_report(true_labels, binary_predictions)
    print("F1 分数:", f1)
    print("F1 分数:\n", F1)

    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    print('\n训练集损失:', train_loss)
    print('训练准确率:', train_accuracy)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('\n测试集损失:', test_loss)
    print('测试准确率:', test_accuracy)

    num_classes = len(np.unique(true_labels))
    p_means = p_mean(true_labels, binary_predictions, num_classes)
    g_means = g_mean(true_labels, binary_predictions, num_classes)

    for i in range(num_classes):
        print(f"类别 {i} 的 P-mean 值: {p_means[i]}")
        print(f"类别 {i} 的 G-mean 值: {g_means[i]}")


train_CNN(x_train_oversampled, y_train_oversampled, x_test, y_test, epochs=10, batch_size=128, seed=50)

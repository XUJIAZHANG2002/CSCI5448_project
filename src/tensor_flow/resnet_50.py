# tf_resnet50.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------------------
# Bottleneck Block (ResNet-50)
# -----------------------------------------
class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, planes, stride=1):
        super().__init__()

        self.conv1 = layers.Conv2D(planes, 1, use_bias=False)
        self.bn1   = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(
            planes, 3, strides=stride, padding="same", use_bias=False
        )
        self.bn2   = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(
            planes * self.expansion, 1, use_bias=False
        )
        self.bn3   = layers.BatchNormalization()

        self.shortcut = None
        if stride != 1:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(planes * self.expansion, 1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, x):
        identity = x

        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.shortcut:
            identity = self.shortcut(x)

        out = tf.nn.relu(out + identity)
        return out


# -----------------------------------------
# ResNet-50
# -----------------------------------------
class ResNet50(Model):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)
        self.bn1   = layers.BatchNormalization()

        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # 4 ResNet stages
        self.layer1 = self._make_layer(64,  3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.gap = layers.GlobalAveragePooling2D()
        self.fc  = layers.Dense(num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers_list = [Bottleneck(planes, stride)]
        for _ in range(1, blocks):
            layers_list.append(Bottleneck(planes))
        return layers_list

    def call(self, x):
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in self.layer1: x = layer(x)
        for layer in self.layer2: x = layer(x)
        for layer in self.layer3: x = layer(x)
        for layer in self.layer4: x = layer(x)

        x = self.gap(x)
        return self.fc(x)


# -----------------------------------------
# X-test
# -----------------------------------------
if __name__ == "__main__":
    model = ResNet50(1000)
    x = tf.random.normal((1, 224, 224, 3))
    out = model(x)
    print("TF ResNet-50 output shape:", out.shape)

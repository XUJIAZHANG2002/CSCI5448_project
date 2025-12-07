# tf_resnet50.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------------------------------------
# Bottleneck Block
# ---------------------------------------------------------
class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1):
        super().__init__()

        # 1x1 reduction
        self.conv1 = layers.Conv2D(planes, 1, use_bias=False)
        self.bn1   = layers.BatchNormalization()

        # 3x3 conv
        self.conv2 = layers.Conv2D(
            planes, 3, strides=stride, padding="same", use_bias=False
        )
        self.bn2   = layers.BatchNormalization()

        # 1x1 expansion
        self.conv3 = layers.Conv2D(
            planes * self.expansion, 1, use_bias=False
        )
        self.bn3   = layers.BatchNormalization()

        # ---------------------------------------------------------
        # Shortcut logic:
        # Projection needed when spatial or channel dims don't match
        # ---------------------------------------------------------
        self.shortcut = None
        if stride != 1 or in_channels != planes * self.expansion:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(
                    planes * self.expansion,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False
                ),
                layers.BatchNormalization()
            ])

    def call(self, x, training=False):
        identity = x

        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)

        if self.shortcut:
            identity = self.shortcut(x, training=training)

        out = tf.nn.relu(out + identity)
        return out


# ---------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------
class ResNet50(Model):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)
        self.bn1   = layers.BatchNormalization()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Track channels flowing through the network
        self.in_channels = 64

        # Build 4 main ResNet stages
        self.layer1 = self._make_layer(planes=64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        # Classification head
        self.gap = layers.GlobalAveragePooling2D()
        self.fc  = layers.Dense(num_classes)

    # Build a ResNet layer (stage)
    def _make_layer(self, planes, blocks, stride):
        layers_list = []

        # First block may downsample
        layers_list.append(Bottleneck(
            in_channels=self.in_channels,
            planes=planes,
            stride=stride
        ))
        # After first block, channels become planes * expansion
        self.in_channels = planes * Bottleneck.expansion

        # Remaining blocks keep same channels
        for _ in range(1, blocks):
            layers_list.append(Bottleneck(
                in_channels=self.in_channels,
                planes=planes,
                stride=1
            ))

        return layers_list

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.maxpool(x)

        for layer in self.layer1: x = layer(x, training=training)
        for layer in self.layer2: x = layer(x, training=training)
        for layer in self.layer3: x = layer(x, training=training)
        for layer in self.layer4: x = layer(x, training=training)

        x = self.gap(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":
    model = ResNet50(num_classes=1000)

    x = tf.random.normal((1, 224, 224, 3))
    out = model(x)
    print("Output shape:", out.shape)   # (1, 1000)

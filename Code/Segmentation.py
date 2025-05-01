import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# T-BLOCK + ATTENTION MODULE
# -----------------------------
class TBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dilation_rate=2, name_prefix="T_block"):
        super(TBlock, self).__init__(name=name_prefix)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name_prefix}_conv1')
        self.conv2 = layers.Conv2D(filters, 3, dilation_rate=dilation_rate, padding='same', activation='relu', name=f'{name_prefix}_conv2')
        self.concat = layers.Concatenate(name=f'{name_prefix}_concat')
        self.bn = layers.BatchNormalization(name=f'{name_prefix}_bn')
        self.relu = layers.ReLU(name=f'{name_prefix}_relu')

    def call(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        concat_out = self.concat([conv1_out, conv2_out])
        bn_out = self.bn(concat_out)
        return self.relu(bn_out)

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, name_prefix="channel_attention"):
        super(ChannelAttention, self).__init__(name=name_prefix)
        self.ratio = ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.shared_dense_1 = layers.Dense(self.filters // self.ratio, activation='relu', name=f'{self.name}_dense1')
        self.shared_dense_2 = layers.Dense(self.filters, activation='sigmoid', name=f'{self.name}_dense2')

    def call(self, x):
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)

        avg_out = self.shared_dense_2(self.shared_dense_1(tf.expand_dims(tf.expand_dims(avg_pool, 1), 1)))
        max_out = self.shared_dense_2(self.shared_dense_1(tf.expand_dims(tf.expand_dims(max_pool, 1), 1)))

        attention = layers.Add()([avg_out, max_out])
        return layers.Multiply(name=f'{self.name}_multiply')([x, attention])

# -----------------------------
# DECODER BLOCK
# -----------------------------
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, name_prefix="decoder_block"):
        super(DecoderBlock, self).__init__(name=name_prefix)
        self.upconv = layers.Conv2DTranspose(filters, 3, strides=2, padding='same', name=f'{name_prefix}_upconv')
        self.concat = layers.Concatenate(name=f'{name_prefix}_concat')
        self.tblock = TBlock(filters, name_prefix=f'{name_prefix}_Tblock')

    def call(self, x, skip):
        x = self.upconv(x)
        x = self.concat([x, skip])
        return self.tblock(x)

# -----------------------------
# T-NET ARCHITECTURE
# -----------------------------
class TNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(TNet, self).__init__()
        self.inputs_ = layers.Input(shape=input_shape)

        # Encoder
        self.enc1 = TBlock(64, name_prefix='enc1')
        self.pool1 = layers.MaxPooling2D((2, 2), name='pool1')

        self.enc2 = TBlock(128, name_prefix='enc2')
        self.pool2 = layers.MaxPooling2D((2, 2), name='pool2')

        self.enc3 = TBlock(256, name_prefix='enc3')
        self.pool3 = layers.MaxPooling2D((2, 2), name='pool3')

        self.enc4 = TBlock(512, name_prefix='enc4')
        self.pool4 = layers.MaxPooling2D((2, 2), name='pool4')

        # Bottleneck
        self.bottleneck = TBlock(1024, name_prefix='bottleneck')
        self.attn = ChannelAttention(name_prefix='bottleneck_attn')

        # Decoder
        self.dec1 = DecoderBlock(512, name_prefix='dec1')
        self.dec2 = DecoderBlock(256, name_prefix='dec2')
        self.dec3 = DecoderBlock(128, name_prefix='dec3')
        self.dec4 = DecoderBlock(64, name_prefix='dec4')

        self.out_conv = layers.Conv2D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax', name='output')

    def call(self, inputs):
        x1 = self.enc1(inputs)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        b = self.bottleneck(p4)
        b = self.attn(b)

        d1 = self.dec1(b, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)

        return self.out_conv(d4)

# -----------------------------
# UNET & VNET FOR COMPARISON
# -----------------------------
class UNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(UNet, self).__init__()
        self.encoder = [
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
        ]
        self.middle = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.decoder = [
            layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
        ]
        self.out = layers.Conv2D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax')

    def call(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, layers.Conv2D):
                skips.append(x)

        x = self.middle(x)

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(skips):
                x = layers.Concatenate()([x, skips[-(i+1)]])

        return self.out(x)

class VNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(VNet, self).__init__()
        self.encoder = [
            layers.Conv3D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling3D(),
            layers.Conv3D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling3D(),
        ]
        self.middle = layers.Conv3D(64, 3, padding='same', activation='relu')
        self.decoder = [
            layers.Conv3DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv3DTranspose(16, 3, strides=2, padding='same', activation='relu'),
        ]
        self.out = layers.Conv3D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax')

    def call(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, layers.Conv3D):
                skips.append(x)

        x = self.middle(x)

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(skips):
                x = layers.Concatenate()([x, skips[-(i+1)]])

        return self.out(x)

# -----------------------------
# INSTANTIATION EXAMPLE
# -----------------------------
input_shape = (128, 128, 1)
num_classes = 1

model_tnet = TNet(input_shape, num_classes)
model_unet = UNet(input_shape, num_classes)
model_vnet = VNet((64, 128, 128, 1), num_classes)  # 3D input for VNet

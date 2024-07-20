import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Multiply, Add, Activation, MaxPooling1D, Concatenate, Input

# Convolved Normalized Pooled (CNPM) Block
def cnpm_block(inputs, filters, kernel_size=5, pool_size=2):
    x = Conv1D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # x = MaxPooling1D(pool_size)(x)
    return x

# Identity Block
def identity_block(inputs, filters, kernel_size):
    x = Conv1D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Basic Block
def basic_block(inputs, filters, kernel_size):
    x = Conv1D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    # x = ReLU()(x)
    return x

# Identity and Basic (IDBN) Block
def identity_basic_block(inputs, filters, kernel_sizes=[3, 5, 7]):
    idb_outputs = []

    for kernel_size in kernel_sizes:
        id_output = identity_block(inputs, filters, kernel_size)
        id_output = identity_block(id_output, filters, kernel_size)
        basic_output = basic_block(id_output, filters, kernel_size)
        combined_output = Add()([inputs, basic_output])
        combined_output = ReLU()(combined_output)
        idb_outputs.append(combined_output)

    if len(idb_outputs) > 1:
        idb_combined = Add()(idb_outputs)
    else:
        idb_combined = idb_outputs[0]
    
    return idb_combined

# Channel Attention Block
def channel_attention(inputs, reduction_ratio=16):
    channel = inputs.shape[-1]
    
    # Global Average Pooling
    channel_avg_pool = GlobalAveragePooling1D()(inputs)
    channel_max_pool = GlobalMaxPooling1D()(inputs)

    shared_dense_1 = Dense(channel // reduction_ratio, activation='relu')
    shared_dense_2 = Dense(channel, activation='sigmoid')

    avg_out = shared_dense_2(shared_dense_1(channel_avg_pool))
    max_out = shared_dense_2(shared_dense_1(channel_max_pool))

    # Combine the outputs of average and max pooling
    combined_out = Add()([avg_out, max_out])
    combined_out = tf.expand_dims(combined_out, axis=1)

    # Apply the attention weights to the input features
    channel_attention_output = Multiply()([inputs, combined_out])
    return channel_attention_output

# Spatial Attention Block
def spatial_attention(inputs):
    avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

    concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_attention = Conv1D(1, 7, padding='same', activation='sigmoid')(concat)

    # Apply the attention weights to the input features
    spatial_attention_output = Multiply()([inputs, spatial_attention])
    return spatial_attention_output

# Channel and Spatial Attention (CASb) Block
def channel_spatial_attention_block(inputs, reduction_ratio=16):
    x = channel_attention(inputs, reduction_ratio)
    x = spatial_attention(x)
    return x

# WISNet Block
def wisnet_block(inputs, filters, kernel_sizes=[3, 5, 7], reduction_ratio=16):
    x = identity_basic_block(inputs, filters, kernel_sizes)
    x = channel_spatial_attention_block(x, reduction_ratio)
    return x

# Building the WISNet Model
def build_wisnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNPM Block
    y = cnpm_block(inputs, 64, kernel_size=5)

    # IDBN1 Block
    x = identity_basic_block(y, 64, kernel_sizes=[3, 5, 7])

    # CASb Block
    x = channel_spatial_attention_block(x, reduction_ratio=16)
    x = Add()([x, y])

    # IDBN2 Block
    x = identity_basic_block(x, 64, kernel_sizes=[3, 5, 7])

    # CASb Block
#     x = channel_spatial_attention_block(x, reduction_ratio=16)

    # Global Pooling and Output Layer
    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model


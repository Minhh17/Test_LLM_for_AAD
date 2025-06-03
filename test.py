import tensorflow as tf

inputs = tf.keras.Input(shape=(None,), dtype='int32')
x = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True)(inputs)
x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128))(x)      # chỉ 1 dòng đổi thành GRU nếu muốn
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, epochs=5, validation_data=val_ds)


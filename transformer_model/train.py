from transformer_model.transformer import *
import matplotlib.pyplot as plt
from preprocess import *
import time

class LearningRateSchdule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self):
        super(LearningRateSchdule, self).__init__()
        self.d_model = tf.cast(transformer_config.d_model, tf.float32)
        self.adam_warm_step = transformer_config.adam_warn_step

    def __call__(self, step):  ## 对应论文中的学习率变化公式
        arg1 = tf.math.rsqrt(step)
        arg2 = step*(self.adam_warm_step ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class TrainConfig:
    def __init__(self):
        self.__build()

    def loss_function(self, real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def __build(self):
        self.loss_obj = self.loss_function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LearningRateSchdule())

class Train:
    def __init__(self, train_config):
        self.train_config = train_config
        self.model = Transformer()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.preprocess = PreProcess()
        checkpoint_path = config.transformer_model_file
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.train_config.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            output = self.model(inputs, targets, config.training)
            loss = self.train_config.loss_obj(targets, output)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.train_config.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(targets, output)

    def process(self):
        start = time.time()

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        train_ask, train_answer, test_ask, test_answer, token = self.preprocess.process()
        train_dataset = tf.data.Dataset.from_tensor_slices((train_ask, train_answer)).shuffle(config.buffer_size).batch(config.batch_size, drop_remainder=True)
        for epoch in range(config.epochs):
            for (batch, (inputs, targets)) in enumerate(train_dataset):
                self.train_step(inputs, targets)
                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                self.train_loss.result(),
                                                                self.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))




if __name__=='__main__':
    train = Train(TrainConfig())
    train.process()
    pass

import tensorflow as tf
from tensorflow import keras

class GradientAccumulator(keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), 
            trainable=False) for v in self.trainable_variables
        ]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        # Unpack the data. 
        images, labels = data

        # start the scope of gradient 
        # Open a GradientTape to record the operations run.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            prim_logit, aux_logit = self(images, training=True)
            # Compute the loss value for this minibatch.
            loss = self.compiled_loss(
                labels, [prim_logit, aux_logit], regularization_losses=self.losses
            )

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
            
        # If n_acum_step reach the n_gradients then we apply accumulated gradients 
        # to update the variables otherwise do nothing
        tf.cond(
            tf.equal(self.n_acum_step, self.n_gradients), 
            self.apply_accu_gradients, 
            lambda: None
        )
        # update metrics
        self.compiled_metrics.update_state(
            labels, [prim_logit, aux_logit]
        )

        return {m.name: m.result() for m in self.metrics}
    
    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_variables[i], dtype=tf.float32)
            )
    
    def test_step(self, data):
        # unpack data 
        images, labels = data
        # inference mode 
        prim_logit, aux_logit = self(images, training=False)
        # Compute the loss
        loss = self.compiled_loss(labels, [prim_logit, aux_logit], regularization_losses=self.losses)
        # update metrics    
        self.compiled_metrics.update_state(labels, [prim_logit, aux_logit])

        return {m.name: m.result() for m in self.metrics}
    
    def build_graph(self):
        x = keras.Input(shape=(self.dim))
        return keras.Model(inputs=[x], outputs=self.call(x))
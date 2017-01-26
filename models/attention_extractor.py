import numpy as np
import tensorflow as tf
from keras.engine import Input
from keras.engine import Layer
from keras.layers import LSTM, Dense, Bidirectional, GRU, Embedding, BatchNormalization, Lambda, initializations
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2

from data.html_writer import HtmlWriter


class AttentionExtractor(object):
    def train(self, train_data, validation_data):
        batch_size = 1
        epoch_count = 50

        self._entity_indices = []
        self.register_entity_indices(train_data)
        self.register_entity_indices(validation_data)

        model = self.create_model()
        x_train, attention_mask_train, y_train = train_data.generate_data()
        x_valid, attention_mask_valid, y_valid = validation_data.generate_data()

        model.fit([x_train, attention_mask_train], y_train, batch_size=batch_size, nb_epoch=epoch_count,
                  validation_data=([x_valid, attention_mask_valid], y_valid))

        self._model = model

    def print_report(self, data, result_file):

        data_samples = data.generate_separated_data()

        correct_count = 0
        processed_count = 0
        writer = HtmlWriter(result_file)
        for i, (x, mask, y) in enumerate(data_samples):
            prediction = self._model.predict([x, mask])[0]
            max_prob_index = np.argmax(prediction)
            print x
            print prediction
            words = data.lines[i]
            print words

            label = data.labels[i]
            result_info = []
            is_question = True
            for word_index, word in enumerate(words):
                if word == "##":
                    is_question = False
                    word = "<br>"

                word = word.replace("www.freebase.com/m/", "")

                if is_question:
                    result_info.append(word)
                    continue

                info = {
                    "text": word,
                    "attention": prediction[word_index]
                }

                if label == word_index:
                    info["label"] = 1.0

                result_info.append(info)

            if not max_prob_index == label:
                writer.write_result_box(result_info, "error")
            else:
                writer.write_result_box(result_info)
                correct_count += 1

            processed_count += 1

        total_count = processed_count + data.skipped_count
        linked_count = processed_count

        total_accuracy = 100.0 * correct_count / total_count
        extraction_accuracy = 100.0 * correct_count / linked_count
        linking_accuracy = 100.0 * linked_count / total_count
        writer.write("<div class='info_box'>")
        writer.write("<h3>Results summary</h3><br>")
        writer.write("<b>Total accuracy: </b>" + "{0:.2f}%".format(total_accuracy))
        writer.write("<br><b>Linking accuracy: </b>" + "{0:.2f}%".format(linking_accuracy))
        writer.write("<br><b>Extraction accuracy: </b>" + "{0:.2f}%".format(extraction_accuracy))

        writer.write("</div>")
        writer.close()

    def train_test(self):
        batch_size = 8
        model = self.create_model()
        x_train, y_train = self.generate_data(100)

        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=50)

    def register_entity_indices(self, data):
        for index in data.entity_indices:
            self._entity_indices.append(index)

    def create_embeddings(self, embedding_dim):
        index_count = 2000

        shape = (index_count, embedding_dim)

        embedding_matrix = np.random.uniform(-1.0, 1.0, size=shape)
        embedding_matrix = np.asarray(embedding_matrix)

        # embedding_matrix = np.zeros((index_count, embedding_dim))
        """
        for i in xrange(index_count):
            if i in self._entity_indices:
                embedding_matrix[i][0] = 1.0
            else:
                embedding_matrix[i][0] = 0.0
        """
        layer = Embedding(index_count, embedding_dim, weights=[embedding_matrix])
        return layer

    def create_model(self):
        embedding_dim = 32
        rnn_dim = 32

        attention_mask = Input(shape=(None,), dtype='float32', name='attention_mask')

        model = Sequential()

        model.add(self.create_embeddings(embedding_dim))
        model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True), merge_mode='concat', name='context_bidir_rnn',
                                input_shape=(None, embedding_dim)))
        model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True), merge_mode='concat', name='context_bidir_rnn2',
                                input_shape=(None, embedding_dim)))

        model.add(Dense(1, activation='linear'))
        model.add(BatchNormalization())
        model.add(AttentionLayer(attention_mask))

        model.inputs.append(attention_mask)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def generate_data(self, sample_count):
        def create_data_sample():
            feature_distance = 3
            min_length = feature_distance
            seq_len = np.random.randint(min_length + feature_distance + 1, 80)

            inputs = []
            labels = []

            correct_label = np.random.randint(min_length, seq_len - feature_distance)

            for i in xrange(seq_len):
                value = 0.0
                if i == correct_label + feature_distance:
                    value = 1.0

                inputs.append(value + 1)
                label_prob = 0.0
                if i == correct_label:
                    label_prob = 1.0

                labels.append(label_prob)

            return inputs, labels

        input_seqs = []
        classes = []
        for _ in xrange(sample_count):
            seq, label = create_data_sample()
            input_seqs.append(seq)
            classes.append(label)

        return pad_sequences(input_seqs), pad_sequences(classes)


class AttentionLayer(Layer):
    def __init__(self, attention_mask=None, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self._attention_mask = attention_mask

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        return input_shape[0:2]

    def call(self, x, mask=None):
        squeezed_x = tf.squeeze(x, squeeze_dims=[2])
        if self._attention_mask is not None:
            squeezed_x = squeezed_x * self._attention_mask

        result = tf.nn.softmax(squeezed_x, name="softmax")
        # result = tf.Print(result, [squeezed_x, result], summarize=200)
        return result

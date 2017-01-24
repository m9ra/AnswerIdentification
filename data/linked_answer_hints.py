import re
import numpy as np

from keras.preprocessing.sequence import pad_sequences


class LinkedAnswerHints(object):
    def __init__(self, source_file, vocabulary_parent=None, use_entity_ordering=False, non_vocabulary_ratio=0.1):
        self._use_entity_ordering = use_entity_ordering

        self.entity_indices = []

        if vocabulary_parent is None:
            self._word_index = {"$UNKNOWN$": 1, "$UNKNOWN_ENTITY$": 2}
            self._update_vocabulary = True
        else:
            self._word_index = vocabulary_parent._word_index
            self._update_vocabulary = True

        with open(source_file) as f:
            lines = f.readlines()

        features = []
        labels = []
        masks = []
        self._features = features
        self._labels = labels
        self._masks = masks
        self.lines = []
        self.labels = []

        vocabulary_lines = len(lines) - non_vocabulary_ratio * len(lines)

        for line in lines:
            if len(line.strip()) == 0:
                continue

            line_words = []
            if vocabulary_lines <= 0:
                self._update_vocabulary = False
            else:
                vocabulary_lines -= 1

            parts = [part.strip() for part in line.split("|")]
            dialog_id = parts[0]
            feature_text = parts[1]
            # feature_text = "Hello mis [http://abce - efg] ghr"
            feature_words = re.findall("([^ \[]+|[\[][^\]]+[\]])", feature_text)
            answer_mid = parts[2]

            feature_vector = []
            mask_vector = []
            label_vector = []

            is_question = True
            has_label_marked = False
            self._entity_index = {}
            label_index = 0
            for i, word in enumerate(feature_words):
                line_words.append(word)
                if answer_mid in word and not is_question and not has_label_marked:
                    # word is an answer
                    label_vector.append(1.0)
                    # ensure the label is marked only once
                    has_label_marked = True
                    label_index = i
                else:
                    label_vector.append(0.0)

                is_question_end = word == "##"
                is_entity_word = word.startswith("[")

                if is_question_end:
                    is_question = False

                if is_question or not is_entity_word:
                    mask_element = 0.0
                else:
                    mask_element = 1.0

                word_index = self.get_word_index(word)
                feature_vector.append(word_index)
                mask_vector.append(mask_element)

            if has_label_marked:
                features.append(feature_vector)
                masks.append(mask_vector)
                labels.append(label_vector)
                self.labels.append(label_index)
                self.lines.append(line_words)

    def generate_data(self):
        return pad_sequences(self._features), pad_sequences(self._masks), pad_sequences(self._labels)

    def generate_separated_data(self):
        samples = []
        for x, m, y in zip(self._features, self._masks, self._labels):
            samples.append((pad_sequences([x]), pad_sequences([m]), pad_sequences([y])))

        return samples

    def get_word_index(self, word):
        is_entity = word.startswith("[")

        if is_entity and self._use_entity_ordering:
            if not word in self._entity_index:
                if not self._update_vocabulary:
                    return 2  # unknown entity
                self._entity_index[word] = len(self._entity_index) + 1

            word = "[" + str(self._entity_index[word])

        if not word in self._word_index:
            if not self._update_vocabulary:
                return 1  # unknown word

            new_index = len(self._word_index) + 1
            if is_entity:
                self.entity_indices.append(new_index)
            self._word_index[word] = new_index

        return self._word_index[word]

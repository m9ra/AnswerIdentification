import zipfile

from models.glove import Glove
from data.linked_answer_hints import LinkedAnswerHints
from models.attention_extractor import AttentionExtractor

zip = zipfile.ZipFile('/media/sf_Shared/glove.6B.zip')
str_table = zip.read("glove.6B.50d.txt")
glove = Glove(str_table)
glove.reduce_dim(10)

train_data = LinkedAnswerHints("../train.qdd_ae", non_vocabulary_ratio=0.4, glove_embeddings=glove, mask=False)
dev_data = LinkedAnswerHints("../dev.qdd_ae", vocabulary_parent=train_data)
test_data = LinkedAnswerHints("../test.qdd_ae", vocabulary_parent=train_data)

extractor = AttentionExtractor()
extractor.train(train_data, dev_data)

extractor.print_report(dev_data, "../dev_report.html")
extractor.print_report(train_data, "../train_report.html")
extractor.print_report(test_data, "../test_report.html")

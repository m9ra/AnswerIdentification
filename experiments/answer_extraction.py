from data.linked_answer_hints import LinkedAnswerHints
from models.attention_extractor import AttentionExtractor

train_data = LinkedAnswerHints("../train.qdd_ae", non_vocabulary_ratio=0.0)
dev_data = LinkedAnswerHints("../test.qdd_ae", vocabulary_parent=train_data)

extractor = AttentionExtractor()
extractor.train(train_data, dev_data)
extractor.print_report(dev_data,"../test_report.html")

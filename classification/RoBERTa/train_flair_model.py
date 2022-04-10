from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.models import TARSClassifier
from flair.embeddings import FlairEmbeddings, PooledFlairEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from torch.optim.sgd import SGD

def train_model(corpus):
    label_dict = corpus.make_label_dictionary(label_type="sentiment")
    document_embeddings = TransformerDocumentEmbeddings('cardiffnlp/twitter-roberta-base-sentiment', fine_tune=True)
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type="sentiment")
    # classifier = TextClassifier.load("/export/home/aneezahm001/IR/public_pretrain/twitter-roberta/final-model.pt")
    trainer = ModelTrainer(classifier, corpus)

    trainer_args = {
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "monitor_train": True,
        
    }

    trainer.fine_tune(
        './fine-tune/hand-augmented/twitter-roberta/',
        learning_rate=5.0e-4,
        mini_batch_size=4,
        max_epochs=15,
        optimizer=SGD,
        **trainer_args,
    )

if __name__ == "__main__":
    print("Loading Corpus")

    data_folder = './data/hand-augmented/'

    column_name_map = {1: "text", 2:"label_topic"}

    corpus: Corpus = CSVClassificationCorpus(
        data_folder, 
        column_name_map,
        skip_header=True, 
        delimiter=',',
        label_type="sentiment",
    )

    print("Sample records are shown below:")
    print(corpus.dev[100])
    print(corpus.train[0])

    print("Starting model training")
    train_model(corpus)

    print("Completed training")
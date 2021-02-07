import os

import tensorflow_hub as hub
import official.nlp.bert.tokenization
import tensorflow.compat.v1 as tf

from transformers import BertConfig, BertTokenizerFast
from official.nlp import bert
from tqdm import tqdm

from the_wizard_express.datasets.triviaqa import TriviaQA
from the_wizard_express.datasets.nq import NQ
from the_wizard_express.datasets.squad import Squad
from the_wizard_express.reader.realm import REALMReader
from the_wizard_express.embedder.realm import REALMEmbedder
from the_wizard_express.config import Config

def test_realm_tokenizer(vocabfile):
    tokenizer = BertTokenizerFast(vocabfile)
    tens_tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=vocabfile,
        do_lower_case=True
    )

    dataset = NQ().get_train_data()

    print("\nTesting the similarity of the Tensorflow & Pytorch tokenizer using NQ")
    for text in tqdm(dataset):
        if (not tens_tokenizer.convert_tokens_to_ids(['[CLS]'] + tens_tokenizer.tokenize(text['question']) + ['[SEP]']) == tokenizer(text['question'])['input_ids']):
            raise Exception("The Tensorflow tokenizer and the Pytorch tokenizer don't have similar matches: {} and {}".format(tens_tokenizer.convert_tokens_to_ids(['[CLS]'] + tens_tokenizer.tokenize(text['question']) + ['[SEP]']), tokenizer(text['question'])['input_ids']))
       
    print("Tokenizer is correctly imported")


def test_realm_similarity_reader():
    reader_config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        type_vocab_size=2,
        intermediate_size=3072,
    )

    # Initialize our Pytorch REALM model
    model = REALMReader.load(reader_config, Config.embedder)

    # Initialize the clean Tensorflow REALM model
    bert_module = hub.load(
        "gs://realm-data/cc_news_pretrained/bert",
        tags=None
    )

    deviation = 0
    deviation1 = 0

    dataset = NQ().get_train_data()

    for batch in tqdm(dataset):
        # Prepare inputs using the same tokenizer
        encoded_inputs = model.tokenizer(
            batch["question"],
            batch["answers"][0],
            return_tensors='pt'
        )
        # Convert Pytorch tensors to Tensorflow tensors 
        # TODO: replace return_tensors with TF and use that
        train_ids_tensor = tf.convert_to_tensor(
            encoded_inputs['input_ids'], dtype=tf.int32, name='input_word_ids')
        train_masks_tensor = tf.convert_to_tensor(
            encoded_inputs['attention_mask'], dtype=tf.int32, name='input_mask')
        train_segments_tensor = tf.convert_to_tensor(
            encoded_inputs['token_type_ids'], dtype=tf.int32, name='input_type_ids')

        # Run TF model
        tf_outputs = bert_module.signatures["tokens"](
            input_ids=train_ids_tensor,
            input_mask=train_masks_tensor,
            segment_ids=train_segments_tensor
        )
        # Run Pytorch model
        pytorch_outputs = model.bert_output(**encoded_inputs)
        sequence_output = pytorch_outputs[0][0].detach().numpy()

        datapoint_result = 0
        for i in sequence_output:
            length = len(i)
            intermediate_results = 0
            for j in range(length):
                intermediate_results += (tf_outputs["sequence_output"][0][0].numpy()[j] - i[j])
            datapoint_result += (intermediate_results / length)
        deviation += (datapoint_result / len(sequence_output))

        length = len(sequence_output[0])
        intermediate_results = 0
        for j in range(length):
            intermediate_results += (tf_outputs["sequence_output"][0][0].numpy()[j] - sequence_output[0][j])
        deviation1 += (intermediate_results / length)

    print("\nTesting resulted in a standard deviation of {}".format(deviation / len(dataset)))
    print("\nTesting resulted in a standard deviation in array [0] of {}".format(deviation1 / len(dataset)))


def test_realm_similarity_retriever():
    retriever_config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        type_vocab_size=2,
        intermediate_size=3072,
    )

    # Convert Pytorch tensors to Tensorflow tensors
    # TODO: replace return_tensors with TF and use that
    train_ids_tensor = tf.convert_to_tensor(
        encoded_inputs['input_ids'], dtype=tf.int32, name='input_word_ids')
    train_masks_tensor = tf.convert_to_tensor(
        encoded_inputs['attention_mask'], dtype=tf.int32, name='input_mask')
    train_segments_tensor = tf.convert_to_tensor(
        encoded_inputs['token_type_ids'], dtype=tf.int32, name='input_type_ids')

    # Initialize our Pytorch REALM model
    model = REALMEmbedder.load(retriever_config, Config.embedder)

    # Initialize the clean Tensorflow REALM model
    bert_module = hub.load(
        "gs://realm-data/cc_news_pretrained/embedder",
        tags=None
    )

    # Prepare inputs using the same tokenizer
    encoded_inputs = model.tokenizer(
        ["What is love ?"],
        ["'What Is Love' is a song recorded by the artist Haddaway"],
        return_tensors='pt'
    )

    # Run TF model
    tf_outputs = bert_module.signatures["tokens"](
        input_ids=train_ids_tensor,
        input_mask=train_masks_tensor,
        segment_ids=train_segments_tensor
    )
    # Run Pytorch model
    pytorch_outputs = model.bert_output(**encoded_inputs)

    print(tf_outputs)

    length = len(pytorch_outputs[0][0][0].detach().numpy())
    results = 0
    for i in range(length):
        results += (tf_outputs["sequence_output"][0][0].numpy()[i] - pytorch_outputs[0][0][0].detach().numpy()[i])

    print((results / length) * 100)
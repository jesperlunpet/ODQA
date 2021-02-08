import os
import tensorflow as tf
import numpy as np
import re
import torch

from .embedder import Embedder
from ..config import Config
from ..corpus.corpus import Corpus

from torch import Tensor, nn
from tensorflow.python.saved_model import load_v1_in_v2
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizerFast
from transformers.modeling_bert import BertOnlyMLMHead, ACT2FN

# Main REALMEmbedder model
class REALMEmbedder(BertPreTrainedModel, Embedder):
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = BertTokenizerFast("../Bert/assets/vocab.txt")

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.cls = BertOnlyMLMHead(config)

        # projected_emb = tf.layers.dense(output_layer, params["projection_size"])
        # projected_emb = tf.keras.layers.LayerNormalization(axis=-1)(projected_emb)
        # if is_training:
        #     projected_emb = tf.nn.dropout(projected_emb, rate=0.1)

        self.dense = nn.Linear(config.hidden_size, 128)
        self.LayerNorm = nn.LayerNorm(128)
        self.projected_emb = nn.Dropout(0.1)

        # No need to init weights, as all are getting imported by REALM
        # self.init_weights()

    # TODO: Forward needs to be tinkered a bit with to use the additional layers
    def forward(
        self,
        input_ids: Tensor,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return outputs[1:]
        return {"pooler_output": pooled_output, "hidden_states": sequence_output, "attentions": outputs.attentions}

        # return NextSentencePredictorOutput(
        #     loss=next_sentence_loss,
        #     logits=seq_relationship_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def bert_output(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def embed_context(self, corpus) -> Corpus:
        return corpus.map(lambda example: {'embeddings': self(**self.tokenizer(example["context"], padding=True, truncation=True, return_tensors="pt").to(Config.device))[0].cpu().detach().numpy()}, batched=True, batch_size=5)

    @staticmethod
    def load_tf_checkpoints(model, config, tf_checkpoint_path):
        print("Building PyTorch model from configuration: {}".format(str(config)))
        print("Converting TensorFlow checkpoint from {}".format(tf_checkpoint_path))
        # Load weights from TF model
        init_vars = load_v1_in_v2.load(tf_checkpoint_path, tags=["train"])
        names = []
        arrays = []
        n_params = 0

        for tf_var in init_vars.variables:
            name = tf_var.name
            print("Loading TF weight {} with shape {}".format(name, tf_var.shape))
            n_params += np.prod(tf_var.shape)
            names.append(name)
            arrays.append(tf_var.numpy())

        for name, array in zip(names, arrays):
            name = re.sub(r"module\/|\:0", "", name).strip()
            name = name.split("/")

            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if any(
                n
                in [
                    "adam_v",
                    "adam_m",
                    "AdamWeightDecayOptimizer",
                    "AdamWeightDecayOptimizer_1",
                    "global_step",
                ]
                for n in name
            ):
                print("Skipping {}".format("/".join(name)))
                continue
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    try:
                        pointer = getattr(pointer, scope_names[0])
                    except AttributeError:
                        print("Skipping {}".format("/".join(name)))
                        continue
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert (
                    pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            print("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        return model

    @staticmethod
    def load(config, checkpoints_path = None):
        if(config is None):
            config = BertConfig()
        cache_path = os.path.join(Config.cache_dir, "embedder.pth.tar")
        # Model already exists
        if(os.path.isfile(cache_path)):
            model = REALMEmbedder(config)
            model.load_state_dict(torch.load(cache_path))
            return model
        # From google and local storage
        elif(tf.io.gfile.exists(checkpoints_path)):
            model = REALMEmbedder.load_tf_checkpoints(REALMEmbedder(config), config, checkpoints_path)
            torch.save(model.state_dict(), cache_path)
            return model
        # Local pytorch model from local storage
        elif(os.path.isfile(checkpoints_path)):
            model = REALMEmbedder(config)
            model.load_state_dict(torch.load(checkpoints_path))
            return model
        else:
            raise IOError("Error - No checkpoints found for embedder model")
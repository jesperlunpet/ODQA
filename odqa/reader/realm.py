import os
import tensorflow as tf
import numpy as np
import re
import torch

from torch import nn
from tensorflow.python.saved_model import load_v1_in_v2
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizerFast
from transformers.modeling_bert import BertOnlyMLMHead, ACT2FN
from tokenizers import BertWordPieceTokenizer

from ..config import Config
from .reader import Reader

# Main REALMReader model
class REALMReader(BertPreTrainedModel, Reader):
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = BertTokenizerFast("../Bert/assets/vocab.txt")

        self.bert = BertModel(config)

        self.cls = BertOnlyMLMHead(config)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return total_loss, start_logits, end_logits, outputs.hidden_states, outputs.attentions

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
            
    @staticmethod
    def load_tf_checkpoints(self, config, tf_checkpoint_path):
        print("Building PyTorch model from configuration: {}".format(str(config)))
        tf_path = os.path.abspath(tf_checkpoint_path)
        print("Converting TensorFlow checkpoint from {}".format(tf_path))
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
            pointer = self
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
        return self

    @staticmethod
    def load(config, checkpoints_path):
        if(config is None):
            config = BertConfig()
        cache_path = os.path.join(Config.cache_dir, "reader.pth.tar")
        # Model already exists
        if(os.path.isfile(cache_path)):
            model = REALMReader(config)
            model.load_state_dict(torch.load(cache_path))
            model.eval()
            return model
        # From google or local storage
        elif(tf.io.gfile.exists(checkpoints_path)):
            model = REALMReader(config)

            model.load_tf_checkpoints(model, config, checkpoints_path)

            print("Save PyTorch model to {}".format(cache_path))
            torch.save(model.state_dict(), cache_path)

            model.load_state_dict(torch.load(cache_path))

            model.eval()

            return model
        # Local pytorch model from local storage
        elif(os.path.isfile(checkpoints_path)):
            return REALMReader(config).load_state_dict(torch.load(checkpoints_path))
        else:
            raise IOError("Error - No checkpoints found for Reader model")

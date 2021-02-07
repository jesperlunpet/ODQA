import torch
import datetime
import json
import os
import random
import time
import numpy as np
from transformers import Trainer, get_linear_schedule_with_warmup, AdamW
from ..validation.squad import Squad
from ..validation.accuracy import Accuracy
from torch.nn import CrossEntropyLoss
from ..config import Config

class QATrainer():
    def __init__(self, model, tokenizer, train_data, validation_data, bert=False):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.validation_data = validation_data
        self.squad = Squad()
        self.accuracy = Accuracy()
        self.bert = bert

        # Set the seed value all over the place to make this reproducible.
        self.seed_val = 10
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)
        # Training and validation loss, validation accuracy, and timings.
        self.training_stats = []
        # Measure the total training time for the whole run.
        self.total_t0 = time.time()

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # Adam with 'Weight Decay fix', see: https://arxiv.org/pdf/1711.05101.pdf
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        # Number of training epochs. The BERT authors recommend between 2 and 4.
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        self.epochs = 4

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=1237, num_training_steps=len(self.train_data) * self.epochs
        )

    def compute_loss(self, start_logits, end_logits, start_label, end_label):
        start_label = torch.LongTensor([start_label]).to('cuda:0')
        end_label = torch.LongTensor([end_label]).to('cuda:0')

        ignored_index = start_logits.view(1,-1).size(1)
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

        start_loss = loss_fct(start_logits.view(1,-1), start_label)
        end_loss = loss_fct(end_logits.view(1,-1), end_label)

        # Optimize to rely more on the start_loss than the end_loss
        total_loss = (start_loss * 0.66 + end_loss * 0.34)
        return total_loss

    def _training(self):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        self.model.train()

        dataset_len = len(self.train_data)
        # For each batch of training data...
        for step, batch in enumerate(self.train_data):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

                # Report progress.
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, dataset_len, elapsed
                    )
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: question and context inputs
            #   [1]: attention masks
            #   [2]: start position of answer in context
            #   [3]: end position of answer in context

            # inputs = self.tokenizer(
            #     questions=batch["question"],
            #     titles=batch["title"],
            #     texts=batch["context"],
            #     return_tensors='pt'
            # ).to(device)
            try:
                inputs = {"input_ids": torch.tensor([batch["input_ids"]]).to(Config.device), "attention_mask": torch.tensor([batch["attention_mask"]]).to(Config.device)}
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                if self.bert:
                    start_logits, end_logits = self.model(**inputs)
                    loss = self.compute_loss(start_logits, end_logits, batch["start_positions"], batch["end_positions"])
                else:
                    start_logits, end_logits, relevance_logits = self.model(**inputs)
                    # Get the index of the result that the reader is "most sure of"
                    index = torch.argmax(relevance_logits)

                    # Accumulate the training loss over all of the batches so that we can
                    # calculate the average loss at the end. `loss` is a Tensor containing a
                    # single value; the `.item()` function just returns the Python value
                    # from the tensor.
                    loss = self.compute_loss(start_logits[index], end_logits[index], batch["start_positions"], batch["end_positions"])
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()

            except:
                print(f"Step {step} with question {batch['question']} gave an error")

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / dataset_len

        # Measure how long this epoch took.
        training_time = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        return avg_train_loss, training_time

    def _validation(self):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("\nValidation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables
        total_eval_squad_accuracy = 0
        total_eval_index_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        dataset_len = len(self.validation_data)
        # For each batch of validation data...
        for step, batch in enumerate(self.validation_data):

            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

                # Report progress.
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, dataset_len, elapsed
                    )
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: start position of answer in context
            #   [3]: end position of answer in context

            # inputs = self.tokenizer(
            #     questions=batch["question"],
            #     titles=batch["title"],
            #     texts=batch["context"],
            #     padding=True,
            #     truncation=True,
            #     return_tensors='pt'
            # ).to(device)

            try:
                inputs = {"input_ids": torch.tensor([batch["input_ids"]]).to(Config.device), "attention_mask": torch.tensor([batch["attention_mask"]]).to(Config.device)}

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates question, titles and context (sentences).
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    if self.bert:
                        start_logits, end_logits = self.model(**inputs)
                    else:
                        start_logits, end_logits, relevance_logits = self.model(
                            **inputs
                        )

                answer_start = torch.argmax(start_logits)  # Get the most likely beginning of answer with the argmax of the score
                answer_end = torch.argmax(end_logits)      # Get the most likely end of answer with the argmax of the score

                if self.bert:
                    loss = self.compute_loss(start_logits, end_logits, batch["start_positions"], batch["end_positions"])
                else:
                    # Get the index of the result that the reader is "most sure of"
                    index = torch.argmax(relevance_logits)
                    loss = self.compute_loss(start_logits[index], end_logits[index], batch["start_positions"], batch["end_positions"])
                total_eval_loss += loss.item()

                golden_candidate = self.tokenizer.decode(inputs["input_ids"][0, answer_start:answer_end].cpu().tolist())
                
                # print(f"Question: {batch['question']}")
                # print(f"Answer: {batch['answers']['text'][0]}")
                # print(f"Golden candidate: {golden_candidate}\n")

                # Squad accuracy
                total_eval_squad_accuracy += self.squad.score(
                    {"id": batch["id"], "prediction_text": golden_candidate},
                    {"id": batch["id"], "answers": batch['answers']}
                )["exact_match"] / 100
                
                # Answer position accuracy
                start_score = self.accuracy.score(answer_start, batch["start_positions"])
                end_score = self.accuracy.score(answer_end, batch["end_positions"])
                if start_score and end_score:
                    total_eval_index_accuracy += start_score["accuracy"]

            except:
                print(f"Step {step} with question {batch['question']} gave an error")

        # Report the final accuracy for this validation run.
        avg__eval_squad_accuracy = total_eval_squad_accuracy / dataset_len
        print("\n  Squad Accuracy: {0:.2f}".format(avg__eval_squad_accuracy))

        avg_eval_index_accuracy = total_eval_index_accuracy / dataset_len
        print("  Index Accuracy: {0:.2f}".format(avg_eval_index_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / dataset_len

        # Measure how long the validation run took.
        validation_time = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        return avg__eval_squad_accuracy, avg_eval_index_accuracy, avg_val_loss, validation_time

    def train(self):
        for epoch_i in range(0, self.epochs):
            print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, self.epochs))
            avg_train_loss, training_time = self._training()

            avg__eval_squad_accuracy, avg_eval_index_accuracy, avg_val_loss, validation_time = self._validation()

            # Record all statistics from this epoch.
            self.training_stats.append(
                {
                    "epoch": epoch_i + 1,
                    "Training Loss": avg_train_loss,
                    "Valid. Loss": avg_val_loss,
                    "Valid. Squad Accur.": avg__eval_squad_accuracy,
                    "Valid. Index Accur.": avg_eval_index_accuracy,
                    "Training Time": training_time,
                    "Validation Time": validation_time,
                }
            )

        print("\nTraining complete!")

        print(self.training_stats)

        print(
            "Total training took {:} (h:mm:ss)".format(
                str(datetime.timedelta(seconds=int(round((time.time() - self.total_t0)))))
            )
        )

        return self.model

import os, shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.utils.cuda import model_to_device
import nltk
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import manifold
from tensorboardX import SummaryWriter

from tqdm import tqdm
import dill
import warnings

from typing import Callable, Union, Dict
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict

from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.trainer import BaseRunner, ClassificationRunner
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import  AdamW, get_linear_schedule_with_warmup
from transformers.optimization import  Adafactor, AdafactorSchedule




class ProtoVerbClassificationRunner(BaseRunner):
    r"""A runner for prototypical verbalizer
    This class is specially implemented for classification.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self,
                 model: PromptForClassification,
                 config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 loss_function: Optional[Callable] = None,
                 id2label: Optional[Dict] = None,
                 ):
        super().__init__(model = model,
                         config = config,
                         train_dataloader = train_dataloader,
                         valid_dataloader = valid_dataloader,
                         test_dataloader = test_dataloader,
                        )
        self.loss_function = loss_function if loss_function else self.configure_loss_function()
        self.id2label = id2label
        self.label_path_sep = config.dataset.label_path_sep

    def configure_loss_function(self,):
        r"""config the loss function if it's not passed."""
        if self.config.classification.loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            return torch.nn.NLLLoss()
        else:
            raise NotImplementedError

    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        # logits = self.model(batch)

        outputs = self.model.prompt_model(batch)
        hidden, _ = self.model.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.model.extract_at_mask(hidden, batch)
        embedding = self.model.verbalizer.head(outputs_at_mask)

        logits = self.model.verbalizer.sim(embedding, self.model.verbalizer.proto)
        pred = torch.argmax(logits, dim=-1)

        return pred.cpu().tolist(), label.cpu().tolist(), embedding.cpu().tolist(), outputs_at_mask.cpu().tolist()

    def inference_epoch_end(self, split, outputs):
        preds = []
        labels = []
        embeddings = []
        embedding_raws = []
        for pred, label, embedding, embedding_raw in outputs:
            preds.extend(pred)
            labels.extend(label)
            embeddings.extend(embedding)
            embedding_raws.extend(embedding_raw)

        np.savez(os.path.join(self.config.logging.path, f"tsne.png"), label=labels, embedding=embeddings, embedding_raw=embedding_raws)
        # label_count = len(set(labels))
        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
        # embeddings_np = np.asarray(embeddings)
        # embeddings_2d = tsne.fit_transform(embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True))
        # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=7, c=labels, 
        #         vmin=np.min(labels), vmax=np.max(labels), cmap='Paired', 
        #         marker='.')
        # plt.savefig(os.path.join(self.config.logging.path, f"tsne.png"))

        # logger.info(f"Start clustering..., label count = {label_count}")
        # clustering = AgglomerativeClustering(n_clusters=label_count, affinity='cosine', linkage="average").fit(np.asarray(embeddings))
        # logger.info(f"ARS = {adjusted_rand_score(labels, clustering.labels_)}")

        # logger.info("Start clustering raw...")
        # clustering_raw = AgglomerativeClustering(n_clusters=label_count, affinity='cosine', linkage="average").fit(np.asarray(embedding_raws))
        # logger.info(f"ARS = {adjusted_rand_score(labels, clustering_raw.labels_)}")

        self.save_results(split, {
            'preds': preds,
            'labels': labels,
            # 'embeddings': embeddings
        })
        self.save_results(split, {
            'proto':  self.model.verbalizer.proto.cpu().tolist()
        })
        if hasattr(self.model.verbalizer.head, "weight"):
            self.save_results(split, {
                'head': self.model.verbalizer.head.weight.cpu().tolist()
            })

        metrics = OrderedDict()
        for metric_name in self.config.classification.metric:
            metric = classification_metrics(preds, labels, metric_name, id2label=self.id2label, label_path_sep=self.label_path_sep)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch['label'])
        return loss

    def prompt_initialize(self):
        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ) and \
            not hasattr(self.inner_model.template, "optimize_to_initialize" ):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader
        elif using_split == "valid":
            dataloader = self.valid_dataloader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                logits = self.model(batch)
            if hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ):
                self.inner_model.verbalizer.optimize_to_initialize()
            if hasattr(self.inner_model.template, "optimize_to_initialize" ):
                self.inner_model.template.optimize_to_initialize()


    def on_fit_start(self):
        """Some initialization works"""
        if self.config.train.train_verblizer != "post" and self.config.train.train_verblizer != "alternate2":
            self.inner_model.verbalizer.train_proto(self.model, self.train_dataloader, self.config.environment.local_rank)

    def fit(self, ckpt: Optional[str] = None):
        self.set_stop_criterion()
        self.configure_optimizers()

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")
        if self.cur_epoch == 0:
            self.on_fit_start()

        for self.cur_epoch in range(self.cur_epoch, self.num_epochs):
            continue_training = self.training_epoch(self.cur_epoch)
            score = self.inference_epoch("validation")
            copy = None
            if self.best_score is None or ((score - self.best_score) >= 0) == self.config.checkpoint.higher_better:
                copy = 'best'
                self.best_score = score
            self.save_checkpoint('last', extra={"validation_metric": score}, copy = copy)
            if continue_training == -1:
                logger.info("Stop training by reaching maximum num_training_steps")
                break
            if self.config.train.train_verblizer == "alternate":
                self.inner_model.verbalizer.train_proto(self.model, self.train_dataloader, self.config.environment.local_rank)
            # TODO
            if self.config.train.train_verblizer == "pre":
                logger.info(self.inner_model.template.soft_vac())

        if self.config.train.train_verblizer == "post":
            self.inner_model.verbalizer.train_proto(self.model, self.train_dataloader, self.config.environment.local_rank)

        return self.best_score
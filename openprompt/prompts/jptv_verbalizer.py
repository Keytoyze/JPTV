import os
from tqdm import tqdm
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.data_utils import InputFeatures
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput

class JPTVVerbalizer(Verbalizer):

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                 lr: Optional[float] = 1e-3,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 head_ckpt_path: Optional[str] = None,
                 pretrain: Optional[bool] = False,
                 freeze_head: Optional[bool] = False,
                 without_head: Optional[bool] = False
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.lr = lr
        self.mid_dim = mid_dim
        self.pretrain = pretrain
        self.epochs = epochs
        self.trained = False
        self.model = model
        self.tokenizer = tokenizer
        self.freeze_head = freeze_head
        self.num_classes2 = self.num_classes
        self.head_ckpt_path = head_ckpt_path

        self.hidden_dims = self.model.config.hidden_size
        self.without_head = without_head
        if without_head:
            self.head = torch.nn.Identity()
            self.mid_dim = self.hidden_dims
        else:
            self.head = torch.nn.Linear(self.hidden_dims, self.mid_dim, bias=False)
        if self.freeze_head == True:
            for p in self.head.parameters():
                p.requires_grad = False
  
        w = torch.empty((self.num_classes2, self.mid_dim))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
        self.optimizer = torch.optim.Adam(self.group_parameters_proto, lr=self.lr)
        self.log_path = os.path.dirname(logger.log_file)
        
    @property
    def group_parameters_proto(self,):
        r"""Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters() if p.requires_grad] + [self.proto]
        else:
            return [self.proto]
    
    def on_label_words_set(self):
        self.generate_parameters()

    def generate_parameters(self) -> List:
        if self.head_ckpt_path is not None:
            if self.without_head:
                logger.info(f"without head, not use {self.head_ckpt_path}")
            else:
                logger.info(f"load head ckpt: {self.head_ckpt_path}")
                self.head.load_state_dict(torch.load(self.head_ckpt_path, map_location='cpu'))

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto)
        return proto_logits

    def process_outputs(self, outputs: Union[torch.Tensor, torch.Tensor], batch: Union[Dict, InputFeatures], **kwargs):
        proto_logits = self.process_hiddens(outputs[0])
        return proto_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))

    def pcl_loss(self, v_ins, ip_loss=True, ii_loss=True):
        # instance-prototype loss

        loss = 0.
        num = v_ins.shape[1]
        if ip_loss:
            sim_mat = torch.exp(self.sim(v_ins, self.proto))
            
            for i in range(num):
                pos_score = torch.diag(sim_mat[:,i,:])
                neg_score = (sim_mat[:,i,:].sum(1) - pos_score)
                loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
            loss = loss / (num * self.num_classes2 * self.num_classes2)

        # instance-instance loss

        loss_ins = 0.
        if ii_loss:
            for i in range(v_ins.shape[0]):
                sim_instance = torch.exp(self.sim(v_ins, v_ins[i]))
                pos_ins = sim_instance[i]
                neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
                loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
            loss_ins = loss_ins / (num * self.num_classes2 * num * self.num_classes2)
        loss = loss + loss_ins

        return loss

    def train_proto(self, model, dataloader, device):
        if self.pretrain == True:
            self.pretrain_proto(model, dataloader, device)
        else:
            self.train_proto_normal(model, dataloader, device)

    def train_proto_normal(self, model, dataloader, device):

        model.eval()
        embeds = [[] for _ in range(self.num_classes2)]
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.to("cuda:{}".format(device)).to_dict()
                outputs = model.prompt_model(batch)
                hidden, _ = self.gather_outputs(outputs)
                outputs_at_mask = model.extract_at_mask(hidden, batch)
                for j in range(len(outputs_at_mask)):
                    label = batch['label'][j]
                    embeds[label].append(outputs_at_mask[j])
            embeds = [torch.stack(e) for e in embeds]
            embeds = torch.stack(embeds)

            x = self.head(embeds)
            x_norm = F.normalize(x, dim=-1)
            x_mean = torch.mean(x_norm, dim=1, keepdim=False).detach() # [C, E]
            x_mean = F.normalize(x_mean, dim=-1) # [C, E]
            self.proto = nn.Parameter(x_mean, requires_grad=True)
            self.optimizer = torch.optim.Adam(self.group_parameters_proto, lr=self.lr)

        loss = 0.
        for epoch in range(self.epochs):
            x = self.head(embeds)
            self.optimizer.zero_grad()
            loss = self.pcl_loss(x, ii_loss=False)
            loss.backward()
            self.optimizer.step()
            logger.info(f"Total epoch: {epoch}. ProtoVerb loss: {loss}.")
        self.trained = True

        try:
            weight_vocab_id = self.model.lm_head(self.head.weight).argmax(dim=-1).cpu().numpy()
            weight_vocabs = [self.tokenizer.decode(x) for x in weight_vocab_id]
            logger.info(",".join(weight_vocabs))
        except:
            pass

    def pretrain_proto(self, model, dataloader, device):
        model.eval()
 
        parameters = list(self.head.named_parameters()) + list(model.prompt_model.template.named_parameters())

        trained_parameters = []
        for n, p in parameters:
            if p.requires_grad:
                logger.info(f"parameters {n}: {p.numel()}")
                trained_parameters.append(p)

        optimizer = torch.optim.Adam(trained_parameters, lr=self.lr)
        
        losses = []
        for epoch in range(1):
            for i, batch in enumerate(tqdm(dataloader)):
                batch = batch.to("cuda:{}".format(device)).to_dict()
                embeds = [[] for _ in range(self.num_classes2)]
                outputs = model.prompt_model(batch)
                hidden, logits = self.gather_outputs(outputs)
                outputs_at_mask = model.extract_at_mask(hidden, batch)
        
                for j in range(len(outputs_at_mask)):
                    label = batch['label'][j]
                    embeds[label].append(outputs_at_mask[j])
                try:
                    embeds = [torch.stack(e) for e in embeds]
                    embeds = torch.stack(embeds)
                except:
                    continue
                if embeds.shape[0] != 4 and embeds.shape[1] != 2:
                    continue
                x = self.head(embeds)
        
                optimizer.zero_grad()
                loss_pcl = self.pcl_loss(x, ip_loss=False)
                loss = loss_pcl
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().item())

                logger.info(f"Total epoch: {i}. ProtoVerb loss: {loss}.")

                if i in [0, 1, 4, 16, 64, 128, 256, 512, 1024, 2048]:
                    torch.save(self.head.state_dict(), os.path.join(self.log_path, f"pretrain_head_{epoch}_{i}.pt"))
                    soft_emb_state_dict = model.prompt_model.template.state_dict().copy()
                    del soft_emb_state_dict['raw_embedding.weight']
                    torch.save(soft_emb_state_dict, os.path.join(self.log_path, f"pretrain_embedding_{epoch}_{i}.pt"))
        self.trained = True

        weight_vocab_id = self.model.lm_head(self.head.weight).argmax(dim=-1).cpu().numpy()
        weight_vocabs = [self.tokenizer.decode(x) for x in weight_vocab_id]
        logger.info("".join(weight_vocabs))

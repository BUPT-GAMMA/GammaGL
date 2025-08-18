"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch

from .base_model import all_gather_with_grad, concat_all_gather
from .translator import TranslatorBase,TranslatorOutput

from transformers import BertTokenizer
from .Qformer import BertConfig, BertLMHeadModel

class GELU(nn.Module):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            return tlx.gelu(x, approximate=self.approximate)
        else:
            return tlx.gelu(x)  



class TranslatorQformerArxiv(TranslatorBase):
    def __init__(
        self,
        config,
        num_features=768,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.bert_dir = config['bert_dir']

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.behavior_proj = nn.Linear(in_features=self.Qformer.config.hidden_size, out_features=embed_dim)
        self.text_proj = nn.Linear(in_features=self.Qformer.config.hidden_size, out_features=embed_dim)

        self.itm_head = nn.Linear(in_features=self.Qformer.config.hidden_size, out_features=2)

        self.temp = nn.Parameter(0.07 * tlx.ones([]))

        self.max_txt_len = max_txt_len
        self.proj = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            GELU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim)
        )

    def init_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(self.bert_dir)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(encoder_config)
        checkpoint = torch.load(self.bert_dir+"/model.pth", map_location=lambda storage, loc: storage)

        Qformer.load_state_dict(checkpoint['model_state_dict'], strict=True)
        query_tokens = nn.Parameter(
            tlx.zeros((1, num_query_token, encoder_config.hidden_size))
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def forward(self, samples):
        behavior_embeds = tlx.expand_dims(samples[1], axis=1)
        text = samples[2]
        behavior_embeds = behavior_embeds.to(self.device)
        behavior_atts = tlx.ones(behavior_embeds.size()[:-1], dtype=tlx.int64).to(behavior_embeds.device)

        query_tokens = self.query_tokens.expand(behavior_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=behavior_embeds,
            encoder_attention_mask=behavior_atts,
            use_cache=True,
            return_dict=True,
        )

        behavior_feats = tlx.l2_normalize(
            self.behavior_proj(query_output.last_hidden_state), axis=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(behavior_embeds.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = tlx.l2_normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), axis=-1
        )

        ###============== Image-text Contrastive ===================###
        behavior_feats_all = concat_all_gather(
            behavior_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim] # torch.Size([8, 32, 256])
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = tlx.matmul(
            behavior_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = tlx.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), behavior_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = 0
        bs = behavior_embeds.size(0)
        targets = tlx.linspace(rank * bs, rank * bs + bs - 1, bs).to(behavior_embeds.device).to(dtype=tlx.int64)

        loss_itc = (
            tlx.losses.softmax_cross_entropy_with_logits(sim_i2t, targets)
            + tlx.losses.softmax_cross_entropy_with_logits(sim_t2i, targets)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        behavior_embeds_world = all_gather_with_grad(behavior_embeds)
        with torch.no_grad():
            weights_t2i = tlx.softmax(sim_t2i, axis=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = tlx.softmax(sim_i2t, axis=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        behavior_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            behavior_embeds_neg.append(behavior_embeds_world[neg_idx])
        behavior_embeds_neg = tlx.stack(behavior_embeds_neg, axis=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = tlx.stack(text_ids_neg, axis=0)
        text_atts_neg = tlx.stack(text_atts_neg, axis=0)

        text_ids_all = tlx.concat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], axis=0
        )  # pos, pos, neg
        text_atts_all = tlx.concat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            axis=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = tlx.ones(query_tokens_itm.size()[:-1], dtype=tlx.int64).to(
            behavior_embeds.device
        )
        attention_mask_all = tlx.concat([query_atts_itm, text_atts_all], axis=1)

        behavior_embeds_all = tlx.concat(
            [behavior_embeds, behavior_embeds_neg, behavior_embeds], axis=0
        )  # pos, neg, pos
        behavior_atts_all = tlx.ones(behavior_embeds_all.size()[:-1], dtype=tlx.int64).to(
            behavior_embeds.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=behavior_embeds_all,
            encoder_attention_mask=behavior_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = tlx.concat(
            [tlx.ones((bs,), dtype=tlx.int64), tlx.zeros((2 * bs,), dtype=tlx.int64)],
            axis=0,
        ).to(behavior_embeds.device)
        loss_itm = tlx.losses.softmax_cross_entropy_with_logits(output=logits, target=itm_labels)
        

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = tlx.ones(query_tokens.size()[:-1], dtype=tlx.int64).to(
            behavior_embeds.device
        )
        attention_mask = tlx.concat([query_atts, text_tokens.attention_mask], axis=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return TranslatorOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=512,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        behavior_embeds = tlx.expand_dims(samples[1], aixs=1).to('cuda:1')

        if not use_nucleus_sampling:
            behavior_embeds = behavior_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        behavior_atts = tlx.ones(behavior_embeds.size()[:-1], dtype=tlx.int64).to(
            behavior_embeds.device
        )

        model_kwargs = {
            "encoder_hidden_states": behavior_embeds,
            "encoder_attention_mask": behavior_atts,
        }

        input_ids = (
            tlx.convert_to_tensor(
            [[self.tokenizer.bos_token_id]] * samples[1].size(0),  # 创建形状为 [batch_size, 1] 的tensor
            dtype=tlx.int64,  
            device=behavior_embeds.device  
            )
        )
        query_tokens = self.query_tokens.expand(behavior_embeds.shape[0], -1, -1).to(behavior_embeds.device)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


    @classmethod
    def from_config(cls, cfg):
        # Behavior
        behavior_length = cfg.get("behavior_length", 384)

        # Text
        max_txt_len = cfg.get("max_txt_len", 32)

        # Q-Former
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        model = cls(
            config=cfg,
            num_features=behavior_length,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model

    def load_from_pretrained(self, url_or_filename):
        if url_or_filename:
            checkpoint = torch.load(url_or_filename, map_location=lambda storage, loc: storage)
            if "model_state_dict" in checkpoint.keys():
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            msg = self.load_state_dict(state_dict, strict=False)

            logging.info("load checkpoint from %s" % url_or_filename)

            return msg

        return

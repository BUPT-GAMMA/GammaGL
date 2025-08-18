"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from typing import List, Optional

import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from .translator import TranslatorBase
from .Qformer import BertConfig, BertLMHeadModel
from .chatglm2 import ChatGLMForConditionalGeneration, ChatGLMTokenizer

IMAGE_TOKEN_ID = 101


class TranslatorCHATGLMArxiv(TranslatorBase):

    def __init__(
        self,
        config,
        num_features=768,
        num_query_token=32,
        chatglm2_model="",
        max_txt_len=2048,
    ):
        super().__init__()
        self.config = config

        chatglm2_model = config['llm_dir']
        self.llm_dir = config['llm_dir']
        self.bert_dir = config['bert_dir']

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, num_features
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.chatglm2_tokenizer = ChatGLMTokenizer.from_pretrained(chatglm2_model, use_fast=False, trust_remote_code=True)


        self.chatglm2_model = ChatGLMForConditionalGeneration.from_pretrained(chatglm2_model)

        for _, param in self.chatglm2_model.named_parameters():
            param.requires_grad = False

        # self.chatglm2_proj = nn.Linear(
        #     in_features=self.Qformer.config.hidden_size, out_features=self.chatglm2_model.config.hidden_size
        # )
        self.chatglm2_proj = torch.nn.Linear(
            in_features=self.Qformer.config.hidden_size, out_features=self.chatglm2_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len

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

    def prepare_lm_input(self, vtokens, text_input: List[str], answer: Optional[List[str]]):
        bsz, nvtoken, _ = vtokens.size()
        tokenizer = self.chatglm2_tokenizer
        device = self.device

        sequences = []
        labels = []
        if answer is None:
            def get_ids():
                for text in text_input:
                    a_ids = [IMAGE_TOKEN_ID] * nvtoken + tokenizer.encode("", add_special_tokens=True)
                    b_ids = tokenizer.encode(text, add_special_tokens=True)
                    yield a_ids, b_ids
        else:
            def get_ids():
                for text, ans in zip(text_input, answer):
                    a_ids = [IMAGE_TOKEN_ID] * nvtoken + tokenizer.encode(text, add_special_tokens=True)
                    b_ids = tokenizer.encode(ans, add_special_tokens=True)
                    yield a_ids, b_ids
        for a_ids, b_ids in get_ids():
            max_caption_length = self.max_txt_len - (len(a_ids) - nvtoken)
            if len(b_ids) > max_caption_length:
                b_ids = b_ids[: max_caption_length]
            input_ids = a_ids + b_ids
            context_length = len(a_ids)
            nvtoken_id = input_ids.index(IMAGE_TOKEN_ID)
            input_ids = tlx.convert_to_tensor(input_ids, dtype=tlx.int64)
            sequences.append(input_ids)
            label = input_ids.detach().clone()
            # -100 is the ignore index is CELoss
            label[:context_length] = -100
            labels.append(label)

        # pad sequences
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
        inputs_embeds = self.chatglm2_model.transformer.embedding.word_embeddings(input_ids)
        inputs_embeds[:, nvtoken_id: nvtoken_id + nvtoken] = vtokens
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        return input_ids, labels, inputs_embeds

    def forward(self, samples):
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(self.device)
        text = samples[2]
        pre_instruction = ['The description of the paper and its cited papers is as follows:' for _ in range(len(text))]
        instruction = ['\nQuestion: Please summarize the topic and content of the paper and its citations in English. Answer:' for _ in range(len(text))]
        device = self.Qformer.bert.device

        multimodal_atts = tlx.ones(multimodal_embeds.size()[:-1], dtype=tlx.int64).to(device)

        query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1).to(device)
        text_Qformer = self.tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        query_atts = tlx.ones(query_tokens.size()[:-1], dtype=tlx.int64).to(device)
       

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=multimodal_embeds,
            encoder_attention_mask=multimodal_atts,
            return_dict=True,
        )
        vtokens = self.chatglm2_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

        input_ids, labels, inputs_embeds = self.prepare_lm_input(
            vtokens=vtokens, text_input=instruction, answer=text
        )

        outputs = self.chatglm2_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=labels,
        )

        loss = outputs.loss

        return {"loss": loss, "vtokens": vtokens, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        samples,
        prompts=[],
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=2048,
        min_length=1,
        top_p=0.8,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.65,
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
        
        device = self.Qformer.bert.device
        multimodal_embeds = samples[1].unsqueeze(dim=1).to(device)
        bs = len(samples[0])
        title = samples[3]
        instruction = [prompts[0]] * bs

        question_prompt_pre = prompts[1].format(title)

        categories = prompts[2]

        question_prompt = prompts[3]
        qustion = question_prompt_pre + categories + question_prompt

        with self.maybe_autocast():
            multimodal_atts = tlx.ones(multimodal_embeds.size()[:-1], dtype=tlx.int64).to(device)
            query_tokens = self.query_tokens.expand(multimodal_embeds.shape[0], -1, -1)
            text_Qformer = self.tokenizer(
                instruction,
                padding='max_length',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=multimodal_embeds,
                encoder_attention_mask=multimodal_atts,
                return_dict=True,
            )

    
            vtokens = self.chatglm2_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

            #first summarize and then answer the question

            #input_ids, labels, inputs_embeds = self.prepare_lm_input(
            #    vtokens=vtokens, text_input=instruction, answer=None
            #)

            #no summarize, straight Q&A
            
            input_ids, labels, inputs_embeds = self.prepare_lm_input(
                vtokens=vtokens, text_input=[qustion], answer=None
            )

            from transformers.generation.utils import LogitsProcessorList
            logits_processor = LogitsProcessorList()
            from .chatglm2.modeling_chatglm import InvalidScoreLogitsProcessor
            logits_processor.append(InvalidScoreLogitsProcessor())

            #gen_kwargs = {
            #    "max_length": max_length,
            #    "min_length": min_length,
            #    "num_beams": 1,
            #    "do_sample": True,
            #    "top_p": top_p,
            #    "temperature": temperature,
            #    "repetition_penalty": repetition_penalty,
            #    "length_penalty": length_penalty,
            #    "logits_processor": logits_processor
            #}
            gen_kwargs = {"max_length": 1024}

            outputs = self.chatglm2_model.generate(input_ids, inputs_embeds=inputs_embeds, **gen_kwargs)
            

            response_output = []
            for i in range(multimodal_embeds.shape[0]):
                outputs_i = outputs.tolist()[i][len(input_ids[i]):]
                response0 = self.chatglm2_tokenizer.decode(outputs_i)
                response0 = self.chatglm2_model.process_response(response0)
                

                #first summarize and then answer the question
                #if len(response0) > max_length - len(qustion) - 1:
                #    response0 = response0[:max_length - len(qustion) - 1]
                #summary_prompt = prompts[4].format(title, response0, qustion)
                #gen_kwargs = {
                #    "max_length": max_length,
                #    "min_length": 100
                #}
                #response2, history = self.chatglm2_model.chat(tokenizer=self.chatglm2_tokenizer,
                #                                            query=summary_prompt,
                #                                            **gen_kwargs)
                #no summarize, straight Q&A
                response_output.append(response0)

            return response_output

    @classmethod
    def from_config(cls, cfg):
        # multimodal
        num_features = cfg.get("num_features", 768)
        # Text
        max_txt_len = cfg.get("max_txt_len", 32)

        # Q-Former
        num_query_token = cfg.get("num_query_token")

        model = cls(
            config=cfg,
            num_features=num_features,
            num_query_token=num_query_token,
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

# import argparse
import torch
import os
from gammagl.utils.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, AutoModelForCausalLM

print('导入成功')
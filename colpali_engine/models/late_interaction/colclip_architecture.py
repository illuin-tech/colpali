from typing import List, Optional

import torch
from torch import nn
from transformers import SiglipModel, SiglipConfig
from transformers import CLIPModel, CLIPConfig
class ColClip(CLIPModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    def __init__(self, config: CLIPConfig):
        super(ColClip, self).__init__(config=config)
        self.dim = 128
        self.text_projection = nn.Linear(self.config.text_config.hidden_size, self.dim, bias = False)
        self.visual_projection = nn.Linear(self.config.vision_config.hidden_size, self.dim, bias = False)
        self.main_input_name = "doc_input_ids"
        self.post_init()
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("inputs_embeds", None)

        outputs =super().forward(*args, output_hidden_states = True,  **kwargs)

        text_last_hidden_states = outputs['text_model_output']['last_hidden_state']
        vision_last_hidden_states = outputs['vision_model_output']['last_hidden_state']

        text_proj = self.text_projection(text_last_hidden_states)
        vision_proj = self.visual_projection(vision_last_hidden_states)

        text_proj = text_proj / text_proj.norm(dim=-1, keepdim=True)
        text_proj = text_proj * kwargs["attention_mask"].unsqueeze(-1)

        vision_proj = vision_proj / vision_proj.norm(dim=-1, keepdim=True)

        return {"text": text_proj, "vision": vision_proj}

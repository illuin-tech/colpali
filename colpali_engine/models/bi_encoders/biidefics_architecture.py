from transformers import Idefics2Model, Idefics2PreTrainedModel


class BiIdefics2(Idefics2PreTrainedModel):
    def __init__(self, config):
        super(BiIdefics2, self).__init__(config=config)
        self.model: Idefics2Model = Idefics2Model(config)
        self.pooling_strategy = "last"
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.model(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        # pooling - last token
        proj = last_hidden_states[:, -1, :]
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj

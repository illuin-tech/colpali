from torch import nn
from .colphi3.modeling_phi3_v import Phi3VModel, Phi3VPreTrainedModel


class ColPhi3(Phi3VPreTrainedModel):
    def __init__(self, config):
        super(ColPhi3, self).__init__(config=config)
        self.model: Phi3VModel = Phi3VModel(config)
        # if self.model.language_model._tied_weights_keys is not None:
        #     self._tied_weights_keys = [
        #         f"model.language_model.{k}" for k in self.model.language_model._tied_weights_keys
        #     ]
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"
        self.post_init()

        # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.custom_text_proj

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.custom_text_proj = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model

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

        outputs = self.model(*args, **kwargs)
        # last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        # last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.last_hidden_state
        proj = self.custom_text_proj(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj

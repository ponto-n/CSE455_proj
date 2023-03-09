from typing import Optional

import torch
from torch import nn

import numpy as np

from transformers import ResNetModel, ResNetPreTrainedModel, AutoConfig
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


def load_init_model():
    model = ResNetForImageRotation.from_pretrained("microsoft/resnet-50")
    return model

# same shape
def rotation_loss(pred, target, loss_type=None):
    if loss_type is None:
        loss_type = nn.MSELoss
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)

    offsets = torch.tensor([[-2 * np.pi, 0, 2 * np.pi]], device=pred.device)
    return loss_type(reduction='none')(pred + offsets, target).min(dim=-1)[0]

class ResNetForImageRotation(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.num_labels = config.num_labels
        self.resnet = ResNetModel(config)
        # classification head
        self.rot_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = torch.remainder(self.rot_classifier(pooled_output), 2 * np.pi)

        loss = None

        assert self.num_labels == 1
        if labels is not None:
            loss_fct = rotation_loss
            loss = loss_fct(logits, labels).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
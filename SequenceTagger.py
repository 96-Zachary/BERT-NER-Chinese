from torchcrf  import CRF
import torch.nn.functional as F
from transformers.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence


class BertForSequenceTagging(BertPreTrainedModel):
	def __init__(self, config):
		super(BertForSequenceTagging, self).__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
				position_ids=None, inputs_embeds=None, head_mask=None):
		input_ids, input_token_starts = input_data
		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds)
		sequence_output = outputs[0]

		# obtain original token representations from sub_words representations (by selecting the first sub_word)
		origin_sequence_output = [
			layer[starts.nonzero().squeeze(1)]
			for layer, starts in zip(sequence_output, input_token_starts)]
		padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
		padded_sequence_output = self.dropout(padded_sequence_output)

		logits = self.classifier(padded_sequence_output)

		outputs = (logits,)
		if labels is not None:
			loss_mask = labels.gt(-1)
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if loss_mask is not None:
				active_loss = loss_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)[active_loss]
				active_labels = labels.view(-1)[active_loss]
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs

log_soft = F.log_softmax
class Bert_CRF(BertPreTrainedModel):
	def __init__(self, config):
		super(Bert_CRF, self).__init__(config)
		self.num_labels = config.num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, self.num_labels)
		self.init_weights()
		self.crf = CRF(self.num_labels, batch_first=True)

	def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
		outputs = self.bert(input_ids, attn_masks)
		sequence_output = outputs[0]
		sequence_output = self.dropout(sequence_output)
		emission = self.classifier(sequence_output)
		attn_masks = attn_masks.type(torch.uint8)
		if labels is not None:
			loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
			return loss
		else:
			prediction = self.crf.decode(emission, mask=attn_masks)
			return prediction
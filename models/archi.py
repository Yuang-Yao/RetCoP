import torch
import torchvision
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, logging
import tensorboard
import math

#########################################
#   model architecture
#########################################
#   visual encoder
class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet']:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        #   pre-train weight
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            self.vision_dim = 2048
            self.model.fc = torch.nn.Identity()

        elif vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        #   projection
        if projection:
            self.out_dim = self.proj_dim
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,bias=proj_bias)
                                                      ,projection=projection, norm=norm)

    def forward(self, pixel_values):
        embed = self.model(pixel_values)
        embed = self.projection_head_vision(embed)
        return embed



#   text encoder
class TextModel(torch.nn.Module):
    def __init__(self, bert_type='/mnt/data/yayao/CPT_data/emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=False,norm=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 112                      #   max input length

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        #   Projection
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        embed = self.projection_head_text(embed)

        last_layer_output = output['hidden_states'][-1]  
        return last_layer_output


#   text encoder
class TextModel0(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 256                            #   max input length

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        #   Projection
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        embed = self.projection_head_text(embed)
        return embed

# MLP  ProjectionLayer
class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()
        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):
        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)        # L2  L2 norm
        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x


#   Multi-head attention mechanism
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = torch.nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = torch.nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        # output:(batch_size*num_heads，，num_hiddens/num_heads)
        # output shape: (batch_size*num_heads，num of queries，num_hiddens/num_heads)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.nn.functional.softmax(scores)
        output = torch.bmm(self.dropout(self.attention_weights), values)

        # output_concat:(batch_size，，num_hiddens)
        # output_concat shape:(batch_size，num of queries，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(torch.nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = torch.nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        # x: batch、times、numhid


class AddNorm(torch.nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln = torch.nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def build_position_encoding(dim, max_len):
    return NNEmbeddingEncoding(dim, max_len)


class NNEmbeddingEncoding(torch.nn.Module):
    def __init__(self, dim, max_len):
        super(NNEmbeddingEncoding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, dim)

    def forward(self, x, start_time=0):
        if isinstance(x, int):
            position_embeddings = self.position_embeddings(torch.tensor([x], dtype=torch.long).cuda())
        elif isinstance(x, torch.Tensor) and x.dim()==1:
            position_embeddings = self.position_embeddings(x)
        else:
            x_size = x.size(1)
            position_ids = torch.arange(x_size, dtype=torch.long, device=x.device) + start_time
            position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class TokenBaseEmbedding(torch.nn.Module):

    def __init__(
        self,
        dim=768,
        vocab_size=49411, # include <BOS>/<EOS>
        **kwargs
    ):
        super(TokenBaseEmbedding, self).__init__()
        kwargs = {
            "dim": 768,
            "vocab_size": 49411
        }

        activation_name = ('none').lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        embeddings_norm = torch.nn.LayerNorm(768)
        kwargs['embeddings_norm'] = embeddings_norm

        embeddings_pos = build_position_encoding(768, 512)
        kwargs['embeddings_pos'] = embeddings_pos

        embeddings_token_type = torch.nn.Embedding(2, 768)
        kwargs['embeddings_token_type'] = embeddings_token_type

        self.embeddings = torch.nn.Embedding(vocab_size, dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop("embeddings_pos", None)
        self.embeddings_token_type = kwargs.pop('embeddings_token_type', None)
        self.embeddings_token_seg = kwargs.pop('embeddings_token_seg', None)
        self.bw_own_embed = kwargs.pop('bw_own_embed', False)
        self.pos_before = kwargs.pop('pos_before', True)


        if self.bw_own_embed:
            # only for debugging
            self.bw_embeddings = copy.deepcopy(self.embeddings)
            self.bw_embeddings_norm = copy.deepcopy(self.embeddings_norm)
            self.bw_embeddings_pos = copy.deepcopy(self.embeddings_pos)
            self.bw_embeddings_token_type = copy.deepcopy(self.embeddings_token_type)
        self.s_token_bias = None



    def set_s_token_bias(self, s_token_bias):
        self.s_token_bias = s_token_bias

    def forward(self, input_ids):

        embeddings = self.embeddings(input_ids)


        if self.s_token_bias is not None:
            # learnable
            embeddings[input_ids == 49410] = embeddings[input_ids == 49410] + self.s_token_bias

        if self.embeddings_pos is not None:
            # print(self.embeddings_pos)
            pos_inputs = input_ids
            position_embeddings = self.embeddings_pos(pos_inputs)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)

        if self.embeddings_token_type is not None:

            embeddings_token_type = self.embeddings_token_type.weight[0].unsqueeze(0).unsqueeze(1)
            embeddings = embeddings + embeddings_token_type.to(embeddings.dtype)

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)
        # print(self.embeddings_pos, self.pos_before)
        if self.embeddings_pos is not None and not self.pos_before:
            # print("1111")

            pos_inputs = input_ids
            position_embeddings = self.embeddings_pos(pos_inputs)
            embeddings = embeddings + position_embeddings.to(embeddings.dtype)
        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)
        return embeddings
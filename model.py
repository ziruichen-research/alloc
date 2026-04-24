import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class PointwiseFeedforward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, exp_conf):
        super().__init__()
        num_ant = exp_conf['num_ant']
        num_car = exp_conf['num_car']
        hidden_dim = exp_conf['res_mlp']['hidden_dim']
        num_blocks = exp_conf['res_mlp']['num_blocks']
        
        self.linear = nn.Linear(num_ant * num_car * 2, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                PointwiseFeedforward(hidden_dim, hidden_dim)
            )
            self.blocks.append(block)
        self.linear_reverse = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        '''
        data: [batch_size, num_ant, num_car, 2], float
        
        Return
        ans: [batch_size, 2], float
        '''
        batch_size = data.shape[0]
        data = data.reshape(batch_size, -1)
        ans = self.linear(data)
        for block in self.blocks:
            ans = ans + block(ans)
        ans = self.linear_reverse(ans)
        return ans

class BaseCNN(nn.Module):
    def __init__(self, exp_conf):
        super().__init__()
        hidden_size = exp_conf['cnn']['hidden_size']
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=hidden_size, kernel_size=(3, 3), padding=1),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=hidden_size, out_channels=2, kernel_size=(3, 3), padding=1),
        )

    def forward(self, *args, **kwargs):
        pass

class CNN(BaseCNN):
    def __init__(self, exp_conf):
        super().__init__(exp_conf)
        num_ant = exp_conf['num_ant']
        num_car = exp_conf['num_car']
        input_size = num_ant * num_car * 2
        self.fc_model = nn.Sequential(
            nn.Linear(input_size // (4 * 4 * 4), input_size // (4 * 4 * 4)),
            nn.LeakyReLU(0.3),
            nn.Linear(input_size // (4 * 4 * 4), 2)
        )

    def forward(self, data):
        '''
        data: [batch_size, num_ant, num_car, 2], float
        
        Return
        ans: [batch_size, 2], float
        '''
        batch_size = data.shape[0]
        data = data.permute(0, 3, 1, 2) # [batch_size, 2, num_ant, num_car]
        ans = self.conv_model(data) # [batch_size, 2, num_ant // (2 * 2 * 2), num_car // (2 * 2 * 2)]
        ans = ans.reshape(batch_size, -1) # [batch_size, input_size // (4 * 4 * 4)]
        ans = self.fc_model(ans) # [batch_size, 2]
        return ans

class MultitaskCNN(BaseCNN):
    def __init__(self, exp_conf):
        super().__init__(exp_conf)
        num_ant = exp_conf['num_ant']
        num_car = exp_conf['num_car']
        input_size = num_ant * num_car * 2
        num_scenarios = len(exp_conf['data_dirs'])
        self.fc_models = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size // (4 * 4 * 4), input_size // (4 * 4 * 4)),
            nn.LeakyReLU(0.3),
            nn.Linear(input_size // (4 * 4 * 4), 2)
        ) for _ in range(num_scenarios)])

    def forward(self, data, k):
        '''
        data: [batch_size, num_ant, num_car, 2], float
        
        Return
        ans: [batch_size, 2], float
        '''
        batch_size = data.shape[0]
        data = data.permute(0, 3, 1, 2) # [batch_size, 2, num_ant, num_car]
        ans = self.conv_model(data) # [batch_size, 2, num_ant // (2 * 2 * 2), num_car // (2 * 2 * 2)]
        ans = ans.reshape(batch_size, -1) # [batch_size, input_size // (4 * 4 * 4)]
        ans = self.fc_models[k](ans) # [batch_size, 2]
        return ans

class MFCNet(nn.Module):
    def __init__(self, exp_conf):
        super().__init__()
        num_ant = exp_conf['num_ant']
        num_car = exp_conf['num_car']
        hidden_size = exp_conf['mfc_net']['hidden_size']
        num_layers = exp_conf['mfc_net']['num_layers']
        self.lstm_model = nn.LSTM(input_size=num_ant * 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc_model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, data):
        '''
        data: [batch_size, num_ant, num_car, 2], float
        
        Return
        ans: [batch_size, num_car, 2], float
        '''
        batch_size = data.shape[0]
        num_car = data.shape[2]
        data = data.permute(0, 2, 1, 3) # [batch_size, num_car, num_ant, 2]
        data = data.reshape(batch_size, num_car, -1) # [batch_size, num_car, num_ant * 2]
        out, (h_n, c_n) = self.lstm_model(data) # [batch_size, num_car, hidden_size]
        ans = self.fc_model(out) # [batch_size, num_car, 2]
        return ans

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.self_attention_h = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attention_x = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feedforward_h = PointwiseFeedforward(embed_dim, hidden_dim)
        self.feedforward_x = PointwiseFeedforward(embed_dim, hidden_dim)
        
        self.norm1_h = nn.LayerNorm(embed_dim)
        self.norm2_h = nn.LayerNorm(embed_dim)
        self.norm1_x = nn.LayerNorm(embed_dim)
        self.norm2_x = nn.LayerNorm(embed_dim)

    def forward(self, h, x, mask=None):
        norm1_h = self.norm1_h(h)
        norm1_x = self.norm1_x(x)
        attn_h, _ = self.self_attention_h(norm1_h, norm1_h, norm1_h, attn_mask=mask)
        attn_x, _ = self.self_attention_x(norm1_h, norm1_h, norm1_x, attn_mask=mask)
        h = h + attn_h
        x = x + attn_x
        norm2_h = self.norm2_h(h)
        norm2_x = self.norm2_x(x)
        ff_h = self.feedforward_h(norm2_h)
        ff_x = self.feedforward_x(norm2_x)
        h = h + ff_h
        x = x + ff_x
        return h, x

class ALLoc(nn.Module):
    def __init__(self, exp_conf):
        super().__init__()
        num_ant = exp_conf['num_ant']
        num_car = exp_conf['num_car']
        embed_dim = exp_conf['alloc']['embed_dim']
        num_heads = exp_conf['alloc']['num_heads']
        hidden_dim = exp_conf['alloc']['hidden_dim']
        depth = exp_conf['alloc']['depth']
        
        self.cur_embed_x = nn.Parameter(torch.FloatTensor(embed_dim), requires_grad=True)
        self.cur_embed_x.data.fill_(0)
        self.nei_embed_x = nn.Parameter(torch.FloatTensor(embed_dim), requires_grad=True)
        self.nei_embed_x.data.fill_(0)
        self.cur_embed_h = nn.Parameter(torch.FloatTensor(embed_dim), requires_grad=False)
        self.cur_embed_h.data.fill_(0)
        self.nei_embed_h = nn.Parameter(torch.FloatTensor(embed_dim), requires_grad=False)
        self.nei_embed_h.data.fill_(0)

        self.input_fc_h = nn.Linear(num_ant * num_car * 2, embed_dim)
        self.input_fc_x = nn.Linear(2, embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerEncoderLayer(embed_dim, num_heads, hidden_dim))
        self.output_fc_x = nn.Linear(embed_dim, 2)

    def forward(self, nei_h, nei_x, cur_h):
        '''
        nei_h: [batch_size, len_nei, num_ant, num_car, 2], float
        nei_x: [batch_size, len_nei, 2], float
        cur_h: [batch_size, len_cur, num_ant, num_car, 2], float
        
        Return
        x: [batch_size, len_cur, 2], float
        '''
        batch_size = nei_h.shape[0]
        len_nei = nei_h.shape[1]
        len_cur = cur_h.shape[1]
        len_total = len_nei + len_cur
        h = torch.cat((nei_h, cur_h), dim=1) # [batch_size, len_total, num_ant, num_car, 2]
        cur_x = torch.mean(nei_x, dim=1, keepdim=True).expand(batch_size, len_cur, 2) # [batch_size, len_cur, 2]
        x = torch.cat((nei_x, cur_x), dim=1) # [batch_size, len_total, 2]
        h = h.reshape(batch_size, len_total, -1) # [batch_size, len_total, num_ant * num_car * 2]
        h = self.input_fc_h(h) # [batch_size, len_total, embed_dim]
        x = self.input_fc_x(x) # [batch_size, len_total, embed_dim]
        nei_embed_h = self.nei_embed_h.expand(len_nei, -1) # [len_nei, embed_dim]
        cur_embed_h = self.cur_embed_h.expand(len_cur, -1) # [len_cur, embed_dim]
        h_embed = torch.cat((nei_embed_h, cur_embed_h), dim=0) # [len_total, embed_dim]
        h = h + h_embed # [batch_size, len_total, embed_dim]
        nei_embed_x = self.nei_embed_x.expand(len_nei, -1) # [len_nei, embed_dim]
        cur_embed_x = self.cur_embed_x.expand(len_cur, -1) # [len_cur, embed_dim]
        x_embed_matrix = torch.cat((nei_embed_x, cur_embed_x), dim=0) # [len_total, embed_dim]
        x = x + x_embed_matrix
        
        mask = torch.zeros((len_total, len_total), dtype=torch.float32, device=nei_h.device) # [len_total, len_total]
        mask[:, len_nei:] = -np.inf
        for layer in self.layers:
            h, x = layer(h, x, mask)
        x = self.output_fc_x(x[:, len_nei:, :])
        return x

class ICLLoc(nn.Module):
    def __init__(self, exp_conf):
        super().__init__()
        num_ant = exp_conf['num_ant']
        num_car = exp_conf['num_car']
        embed_dim = exp_conf['iclloc']['embed_dim']
        num_heads = exp_conf['iclloc']['num_heads']
        hidden_dim = exp_conf['iclloc']['hidden_dim']
        depth = exp_conf['iclloc']['depth']
        max_len_nei = exp_conf['max_len_nei']

        max_len = max_len_nei * 2 + 1
        self.pos_embed = nn.Parameter(torch.zeros(max_len, embed_dim), requires_grad=True)
        self.input_fc_h = nn.Linear(num_ant * num_car * 2, embed_dim)
        self.input_fc_x = nn.Linear(2, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, depth)
        self.output_fc_x =  nn.Linear(embed_dim, 2)

    def forward(self, nei_h, nei_x, cur_h):
        '''
        nei_h: [batch_size, len_nei, num_ant, num_car, 2], float
        nei_x: [batch_size, len_nei, 2], float
        cur_h: [batch_size, 1, num_ant, num_car, 2], float
        
        Return
        out: [batch_size, len_nei * 2 + 1, 2], float
        '''
        batch_size = nei_h.shape[0]
        len_nei = nei_h.shape[1]
        nei_h = nei_h.reshape(batch_size, len_nei, -1) # [batch_size, len_nei, num_ant * num_car * 2]
        nei_h = self.input_fc_h(nei_h) # [batch_size, len_nei, embed_dim]
        cur_h = cur_h.reshape(batch_size, 1, -1) # [batch_size, 1, num_ant * num_car * 2]
        cur_h = self.input_fc_h(cur_h) # [batch_size, 1, embed_dim]
        nei_x = self.input_fc_x(nei_x) # [batch_size, len_nei, embed_dim]

        seq = torch.stack([nei_h, nei_x], dim=2) # [batch_size, len_nei, 2, embed_dim]
        seq = seq.reshape((batch_size, len_nei * 2, -1)) # [batch_size, len_nei * 2, embed_dim]
        seq = torch.cat([seq, cur_h], dim=1) # [batch_size, len_nei * 2 + 1, embed_dim]
        seq = seq + self.pos_embed[:2 * len_nei + 1] # [batch_size, len_nei * 2 + 1, embed_dim]

        mask = torch.full((len_nei * 2 + 1,) * 2, -np.inf, dtype=torch.float32, device=nei_h.device) # [len_nei * 2 + 1, len_nei * 2 + 1]
        mask = torch.triu(mask, diagonal=1)
        seq = self.decoder(src=seq, mask=mask)
        out = self.output_fc_x(seq)
        return out
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE_MY(nn.Module):
    def __init__(self, dim, length):
        super(VAE_MY, self).__init__()

        self.dim = dim
        self.length = length

        self.encoder = nn.Sequential(   #[batch, length * dim]
            nn.Linear(length * dim, dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, length * dim),
        )
        self.fc11 = nn.Linear(dim, dim)
        self.fc12 = nn.Linear(dim, dim)

        self.relu = nn.ReLU()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        init_weights(self.encoder[0])
        init_weights(self.decoder[0])
        init_weights(self.fc11)
        init_weights(self.fc12)

    def encode(self, x):
        x = x.view(-1, self.dim * self.length)
        h1 = self.encoder(x);
        # print("encode h1", h1.size())
        return self.fc11(h1), self.fc12(h1)

    def decode(self, z):
        h2 = self.decoder(z)
        return h2.view(-1, self.length, self.dim)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def vae_loss(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x.view(-1, self.length * self.dim), x.view(-1, self.length * self.dim), size_average=False)
        # BCE = F.binary_cross_entropy(recon_x.view(-1, self.length * self.dim), x.view(-1, self.length * self.dim), size_average=False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE/self.length + KLD

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decode(z)
        vae_l = self.vae_loss(recon_x, x, mu, logvar)

        return z, vae_l

class CML_Bi_Disen8(torch.nn.Module):
    def __init__(self, user_num, item_num, user_feat_num, item_feat_num, item_feat_len, dim_num, margin) -> object:
        super(CML_Bi_Disen8, self).__init__()
        self.num_users = user_num
        self.num_items = item_num
        self.user_feat_num = user_feat_num
        self.item_feat_num = item_feat_num
        self.latent_dim_mlp = dim_num
        self.margin = margin
        self.dropout = 0.0

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        self.embedding_feat_item = torch.nn.Embedding(num_embeddings=item_feat_num, embedding_dim=self.latent_dim_mlp)
        nn.init.xavier_uniform_(self.embedding_feat_item.weight)
        nn.init.zeros_(self.embedding_feat_item.weight[0])
        # self.bi_pool = BiInteraction(self.latent_dim_mlp)

        self.embedding_feat_user = torch.nn.Embedding(num_embeddings=user_feat_num, embedding_dim=self.latent_dim_mlp)
        nn.init.xavier_uniform_(self.embedding_feat_user.weight)
        nn.init.zeros_(self.embedding_feat_user.weight[0])

        self.item_vae = VAE_MY(self.latent_dim_mlp, item_feat_len)

        self.int_emb_item = nn.Parameter(torch.FloatTensor(self.latent_dim_mlp, 1))
        self.pop_emb_item = nn.Parameter(torch.FloatTensor(self.latent_dim_mlp, 1))
        nn.init.xavier_uniform_(self.int_emb_item)
        nn.init.xavier_uniform_(self.pop_emb_item)
        self.int_emb_user = nn.Parameter(torch.FloatTensor(self.latent_dim_mlp, 1))
        self.pop_emb_user = nn.Parameter(torch.FloatTensor(self.latent_dim_mlp, 1))
        nn.init.xavier_uniform_(self.int_emb_user)
        nn.init.xavier_uniform_(self.pop_emb_user)

        self.linear_feat_pop = nn.Linear(self.latent_dim_mlp, self.latent_dim_mlp)
        self.linear_feat_int = nn.Linear(self.latent_dim_mlp, self.latent_dim_mlp)
        init_weights(self.linear_feat_pop)
        init_weights(self.linear_feat_int)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, user_indices, user_feat, pos_item_indices, pos_item_feat, neg_item_indices, neg_item_feat, comp_neg_indices):
        batch_size = user_indices.shape[0]
        feat_size = pos_item_feat.shape[1]
        neg_item_indices = neg_item_indices.squeeze(1)
        neg_item_feat = neg_item_feat.squeeze(1)

        user_embedding = self.embedding_user(user_indices)
        user_feat_embedding = self.embedding_feat_user(user_feat)
        user_fuse_embedding = torch.cat([user_embedding.unsqueeze(1), user_feat_embedding], dim=1)  # [batch, len, dim]
        user_mask = torch.cat([torch.ones([batch_size, 1], dtype=torch.long).to(device), user_feat], dim=1).unsqueeze(-1)

        user_attn_weight_pop = torch.bmm(user_fuse_embedding, self.pop_emb_user.unsqueeze(0).repeat(batch_size, 1, 1))
        user_attn_weight_pop = user_attn_weight_pop.masked_fill(user_mask.eq(0), -np.inf)
        user_attn_weight_pop = nn.Softmax(dim=1)(user_attn_weight_pop)
        user_fuse_pop = (user_fuse_embedding * user_attn_weight_pop).sum(dim=1)
        user_embedding_hot = self.linear_feat_pop(user_fuse_pop)

        user_attn_weight_int = torch.bmm(user_fuse_embedding, self.int_emb_user.unsqueeze(0).repeat(batch_size, 1, 1))
        user_attn_weight_int = user_attn_weight_int.masked_fill(user_mask.eq(0), -np.inf)
        user_attn_weight_int = nn.Softmax(dim=1)(user_attn_weight_int)
        user_fuse_int = (user_fuse_embedding * user_attn_weight_int).sum(dim=1)
        user_embedding_cold = self.linear_feat_int(user_fuse_int)

        pos_item_embedding = self.embedding_item(pos_item_indices)
        pos_item_feat_embedding = self.embedding_feat_item(pos_item_feat)
        pos_item_fuse_embedding = torch.cat([pos_item_embedding.unsqueeze(1), pos_item_feat_embedding], dim=1)
        pos_item_mask = torch.cat([torch.ones([batch_size, 1], dtype=torch.long).to(device), pos_item_feat],
                                  dim=1).unsqueeze(-1)

        pos_item_attn_weight_pop = torch.bmm(pos_item_fuse_embedding,
                                             self.pop_emb_item.unsqueeze(0).repeat(batch_size, 1, 1))
        pos_item_attn_weight_pop = pos_item_attn_weight_pop.masked_fill(pos_item_mask.eq(0), -np.inf)
        pos_item_attn_weight_pop = nn.Softmax(dim=1)(pos_item_attn_weight_pop)
        pos_item_fuse_pop = (pos_item_fuse_embedding * pos_item_attn_weight_pop).sum(dim=1)
        pos_item_embedding_hot = self.linear_feat_pop(pos_item_fuse_pop)

        pos_item_attn_weight_int = torch.bmm(pos_item_fuse_embedding,
                                             self.int_emb_item.unsqueeze(0).repeat(batch_size, 1, 1))
        pos_item_attn_weight_int = pos_item_attn_weight_int.masked_fill(pos_item_mask.eq(0), -np.inf)
        pos_item_attn_weight_int = nn.Softmax(dim=1)(pos_item_attn_weight_int)
        pos_item_fuse_int = (pos_item_fuse_embedding * pos_item_attn_weight_int).sum(dim=1)
        pos_item_embedding_cold = self.linear_feat_int(pos_item_fuse_int)

        neg_item_embedding = self.embedding_item(neg_item_indices)
        neg_item_feat_embedding = self.embedding_feat_item(neg_item_feat)
        neg_item_fuse_embedding = torch.cat([neg_item_embedding.unsqueeze(1), neg_item_feat_embedding], dim=1)
        neg_item_mask = torch.cat([torch.ones([batch_size, 1], dtype=torch.long).to(device), neg_item_feat],
                                  dim=1).unsqueeze(-1)

        neg_item_attn_weight_pop = torch.bmm(neg_item_fuse_embedding,
                                             self.pop_emb_item.unsqueeze(0).repeat(batch_size, 1, 1))
        neg_item_attn_weight_pop = neg_item_attn_weight_pop.masked_fill(neg_item_mask.eq(0), -np.inf)
        neg_item_attn_weight_pop = nn.Softmax(dim=1)(neg_item_attn_weight_pop)
        neg_item_fuse_pop = (neg_item_fuse_embedding * neg_item_attn_weight_pop).sum(dim=1)
        neg_item_embedding_hot = self.linear_feat_pop(neg_item_fuse_pop)

        neg_item_attn_weight_int = torch.bmm(neg_item_fuse_embedding,
                                             self.int_emb_item.unsqueeze(0).repeat(batch_size, 1, 1))
        neg_item_attn_weight_int = neg_item_attn_weight_int.masked_fill(neg_item_mask.eq(0), -np.inf)
        neg_item_attn_weight_int = nn.Softmax(dim=1)(neg_item_attn_weight_int)
        neg_item_fuse_int = (neg_item_fuse_embedding * neg_item_attn_weight_int).sum(dim=1)
        neg_item_embedding_cold = self.linear_feat_int(neg_item_fuse_int)

        ##  CML
        pos_distances_hot = torch.sum(torch.pow(user_embedding_hot - pos_item_embedding_hot, 2), dim=-1)
        neg_distances_hot = torch.sum(torch.pow(user_embedding_hot - neg_item_embedding_hot, 2), dim=-1)
        loss_per_pair_hot = self.relu(pos_distances_hot - neg_distances_hot + self.margin)

        pos_distances_cold = torch.sum(torch.pow(user_embedding_cold - pos_item_embedding_cold, 2), dim=-1)
        neg_distances_cold = torch.sum(torch.pow(user_embedding_cold - neg_item_embedding_cold, 2), dim=-1)
        loss_per_pair_cold = self.relu(pos_distances_cold - neg_distances_cold + self.margin)

        ## norm
        comp_neg_embedding = self.embedding_item(comp_neg_indices)
        hidden_item_vae, item_vae_loss = self.item_vae(pos_item_feat_embedding)
        mmi_loss = -torch.sum(F.logsigmoid((hidden_item_vae * pos_item_embedding).sum(dim=1) - (hidden_item_vae * comp_neg_embedding).sum(dim=1)))
        item_vae_loss = mmi_loss + item_vae_loss + F.mse_loss(hidden_item_vae, pos_item_embedding, size_average=False)
        return loss_per_pair_hot, loss_per_pair_cold, item_vae_loss

    def predict(self, user_indices, item_indices, user_feat, item_feat, c_items):  # [1], [item_size]
        batch_size = item_indices.shape[0]

        user_embedding = self.embedding_user(user_indices)
        user_feat_embedding = self.embedding_feat_user(user_feat)
        user_fuse_embedding = torch.cat([user_embedding.unsqueeze(1), user_feat_embedding], dim=1)  # [batch, len, dim]
        user_mask = torch.cat([torch.ones([1, 1], dtype=torch.long).to(device), user_feat], dim=1).unsqueeze(-1)

        user_attn_weight_pop = torch.bmm(user_fuse_embedding, self.pop_emb_user.unsqueeze(0))
        user_attn_weight_pop = user_attn_weight_pop.masked_fill(user_mask.eq(0), -np.inf)
        user_attn_weight_pop = nn.Softmax(dim=1)(user_attn_weight_pop)
        user_fuse_pop = (user_fuse_embedding * user_attn_weight_pop).sum(dim=1)
        user_embedding_hot = self.linear_feat_pop(user_fuse_pop).repeat(batch_size, 1)

        user_attn_weight_int = torch.bmm(user_fuse_embedding, self.int_emb_user.unsqueeze(0))
        user_attn_weight_int = user_attn_weight_int.masked_fill(user_mask.eq(0), -np.inf)
        user_attn_weight_int = nn.Softmax(dim=1)(user_attn_weight_int)
        user_fuse_int = (user_fuse_embedding * user_attn_weight_int).sum(dim=1)
        user_embedding_cold = self.linear_feat_int(user_fuse_int).repeat(batch_size, 1)

        item_embedding = self.embedding_item(item_indices)
        item_feat_embedding = self.embedding_feat_item(item_feat)
        # vae generate
        hidden_item_vae, _ = self.item_vae(item_feat_embedding)
        item_embedding = hidden_item_vae * c_items.unsqueeze(-1) + item_embedding * (1 - c_items.unsqueeze(-1))  # torch.where(c_items.unsqueeze(-1)>0, hidden_item_vae, item_embedding)
        item_fuse_embedding = torch.cat([item_embedding.unsqueeze(1), item_feat_embedding], dim=1)  # [batch, len, dim]
        item_mask = torch.cat([torch.ones([batch_size, 1], dtype=torch.long).to(device), item_feat], dim=1).unsqueeze(-1)

        item_attn_weight_pop = torch.bmm(item_fuse_embedding, self.pop_emb_item.unsqueeze(0).repeat(batch_size, 1, 1))
        item_attn_weight_pop = item_attn_weight_pop.masked_fill(item_mask.eq(0), -np.inf)
        item_attn_weight_pop = nn.Softmax(dim=1)(item_attn_weight_pop)
        item_fuse_pop = (item_fuse_embedding * item_attn_weight_pop).sum(dim=1)
        item_embedding_hot = self.linear_feat_pop(item_fuse_pop)

        item_attn_weight_int = torch.bmm(item_fuse_embedding, self.int_emb_item.unsqueeze(0).repeat(batch_size, 1, 1))
        item_attn_weight_int = item_attn_weight_int.masked_fill(item_mask.eq(0), -np.inf)
        item_attn_weight_int = nn.Softmax(dim=1)(item_attn_weight_int)
        item_fuse_int = (item_fuse_embedding * item_attn_weight_int).sum(dim=1)
        item_embedding_cold = self.linear_feat_int(item_fuse_int)

        # --------------------------------------------------
        ui_distances_hot = -torch.sum(torch.pow(user_embedding_hot - item_embedding_hot, 2), dim=-1)
        ui_distances_cold = -torch.sum(torch.pow(user_embedding_cold - item_embedding_cold, 2), dim=-1)

        return ui_distances_hot.view(-1), ui_distances_cold.view(-1)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=300, word_gru_hidden_dim=350, sent_gru_hidden_dim=350,
                word_gru_num_layers=2, sent_gru_num_layers=2, word_att_dim=150, sent_att_dim=150, use_layer_norm=False, dropout=0.5):

        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        # classifier
        self.fc = nn.Linear(2 * sent_gru_hidden_dim, num_classes)

        # NOTE MODIFICATION (BUG)
        self.log_softmax = nn.LogSoftmax(dim=-1) # option 1
        # erase this line # option 2

        # NOTE MODIFICATION (FEATURES)
        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        # print(docs.shape)
        # print(doc_lengths.shape)
        # print(sent_lengths.shape)
        doc_embeds, word_att_weights, sent_att_weights = self.sent_attention(docs, doc_lengths, sent_lengths)

        # NOTE MODIFICATION (BUG)
        # scores = self.out(self.fc(doc_embeds)) # option 1
        scores = self.fc(doc_embeds) # option 2
        log_scores = self.log_softmax(scores)

        return scores


class SentenceAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers,
                                            word_att_dim, use_layer_norm, dropout)

        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(2 * word_gru_hidden_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        # NOTE MODIFICATION (FEATURES)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, docs, doc_lengths, sent_lengths):
        # Sort documents by decreasing order in length
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim=0, descending=True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `docs` was of size (num_docs, padded_doc_length, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_sents.batch_sizes

        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_docs, padded_doc_length)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=doc_lengths.tolist(), batch_first=True)

        # Word attention module
        sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)

        # NOTE MODIFICATION (FEATURES)
        sents = self.dropout(sents)

        # Sentence-level GRU over sentence embeddings
        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))

        # NOTE MODIFICATION (FEATURES)
        if self.use_layer_norm:
            normed_sents = self.layer_norm(packed_sents.data)
        else:
            normed_sents = packed_sents

        # Sentence attention

        att = torch.tanh(self.sent_attention(normed_sents.data))
        att = self.sentence_context_vector(att).squeeze(1)

        # NOTE MODIFICATION (BUG)
        val = att.max()
        att = torch.exp(att - val)

        # Restore as documents by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        # Note MODIFICATION (BUG)
        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as documents by repadding
        docs, _ = pad_packed_sequence(packed_sents, batch_first=True)

        # Compute document vectors
        docs = docs * sent_att_weights.unsqueeze(2)
        docs = docs.sum(dim=1)

        # Restore as documents by repadding
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)

        # Restore the original order of documents (undo the first sorting)
        _, doc_unperm_idx = doc_perm_idx.sort(dim=0, descending=False)
        docs = docs[doc_unperm_idx]

        # NOTE MODIFICATION (BUG)
        word_att_weights = word_att_weights[doc_unperm_idx]
        sent_att_weights = sent_att_weights[doc_unperm_idx]

        return docs, word_att_weights, sent_att_weights


class WordAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # output (batch, hidden_size)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout)

        # NOTE MODIFICATION (FEATURES)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight.requires_grad = not freeze

    def forward(self, sents, sent_lengths):
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents)
        sents = self.dropout(sents)

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings
        packed_words, _ = self.gru(packed_words)

        # NOTE MODIFICATION (FEATURES)
        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        # Word Attenton
        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val) # att.size: (n_words)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        # NOTE MODIFICATION (BUG) : attention score sum should be in dimension
        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Compute sentence vectors
        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        # NOTE MODIFICATION BUG
        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights

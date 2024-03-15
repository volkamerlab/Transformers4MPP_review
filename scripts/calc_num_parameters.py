"""
the intuition and calculations of this script were made possible by the blog posts from Jay Alammar and Dmytro Nikolaiev
https://jalammar.github.io/illustrated-gpt2/#part-3-beyond-language-modeling
https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0
"""
import os

import pandas as pd


def calc_model_params(d_model, n_layers, enc: bool = True, autoreg: bool = False):
    # 1- Each attention-head is composed of three Q, K, V matrices that need to be learnt via gradient update. Each
    #    matrix is of size d_model*d_model.
    # 2- when there are multi-head attention (MHA), the original input is split between these heads, and each Q_i,
    #    K_i, and V_i becomes of size d_model*d_split, d_split = d_model / num_heads.
    # 3- So, regardless of how many heads are there, each MHA will need to learn d_model*d_model weights for each
    #    set of matrices alongside the bias terms which will equal d_model for each matrix
    # 4- Additionally, there is a projection matrix that maps the output of the attention head(s) to original space,
    #    which accounts for another d_model*d_model weights and d_model bias
    # Therefore, for a single or multi-head attention, the equation becomes as follows
    weights_per_mat = d_model * d_model
    biases_per_mat = d_model
    num_mats = 4  # Q, K, V, and Projection
    multi_head_attn = num_mats * (weights_per_mat + biases_per_mat)

    # After each attention module, comes an add & normalization module that are also learned by the models
    # 1- an addition vector and a scaling vector are learned and added to each token. These vectors are meant for the
    #    stabilization of the calculation from the different modules
    # 2- each vector is of the same size as the token size. Therefore, each add & norm layer account for 2 vectors that
    #    each requires d_model parameters
    add_params = d_model
    norm_params = d_model
    add_n_norm = add_params + norm_params

    # 1- The feed forward network is traditionally composed of two layers. The first one is usually 4 times the model
    #    dimension (i.e., d_ff1 = 4*d_model), and the second one maps back to the model dimension
    #    (i.e., d_ff2 = d_model)
    # 2- Therefore, this module requires d_model * d_ff1 weights and d_ff1 biases for the first layer, and d_ff1 * d_ff2
    #    weights and d_ff2 biases for the second layer
    weights_first_layer = d_model * (4 * d_model)
    biases_first_layer = 4 * d_model
    weights_second_layer = (4 * d_model) * d_model
    biases_second_layer = d_model
    ffn = (weights_first_layer + biases_first_layer) + (weights_second_layer + biases_second_layer)

    # An encoder layer consists of one MHA, two add & norm, and one FFN. The same is true for an autoregressive model.
    encoder_params = multi_head_attn + (2 * add_n_norm) + ffn

    # A decoder layer consists of two attention modules, three add & norm, and one ffn.
    decoder_params = (2 * multi_head_attn) + (3 * add_n_norm) + ffn

    # For more than one layer of an encoder/decoder, we multiply each modul params by this number
    if n_layers is not None:
        enc_dec_params = n_layers * (encoder_params + decoder_params)
        enc_only_params = n_layers * encoder_params
    else:
        enc_dec_params = None
        enc_only_params = None
    if enc or autoreg:
        return enc_only_params
    else:
        return enc_dec_params


def calc_embed_params(d_model, vocab_size):
    # these parameters are determined by the number of tokens (vocabulary) that need to be learned by the model and the
    # embedding dimension that is assigned for each token
    return d_model * vocab_size


def calc_pos_embed_params(d_model, seq_len):
    # a vector of the same size as the token's dimension (i.e., d_model) is needed for each token in the sequence to let
    # the model now this token's position.
    # Therefore, the determining factor of positional embedding parameters are sequence length and model dimensions
    return d_model * seq_len


def sum_all_params(d_model, n_layers, vocab_size, seq_len, enc: bool = True, autoreg: bool = False):
    model_params = calc_model_params(d_model, n_layers, enc=enc, autoreg=autoreg)

    if vocab_size is not None:
        embed_params = calc_embed_params(d_model, vocab_size)
    else:
        embed_params = None

    if seq_len is not None:
        pos_embed_params = calc_pos_embed_params(d_model, seq_len)
    else:
        pos_embed_params = None

    if model_params is not None and embed_params is not None and pos_embed_params is not None:
        all_params = round((model_params + embed_params + pos_embed_params) / 1000000, 0)
    else:
        all_params = None

    return round(model_params / 1000000, 0), round(embed_params/1000, 0), pos_embed_params, all_params


def process_data(df, data_path):
    model = []
    embed = []
    pos = []
    all_added = []
    for _, row in df.iterrows():
        d_model = row.loc['d_model']
        n_layers = row.loc['n_layers']
        vocab_size = row.loc['vocab_size']
        seq_len = row.loc['seq_len']
        enc = row.loc['enc']
        autoreg = row.loc['autoreg']
        model_params, embed_params, pos_embed_params, all_params = sum_all_params(d_model,
                                                                                  n_layers,
                                                                                  vocab_size,
                                                                                  seq_len,
                                                                                  enc,
                                                                                  autoreg)
        model.append(model_params)
        embed.append(embed_params)
        pos.append(pos_embed_params)
        all_added.append(all_params)

    calculated_params = pd.DataFrame({'model': df.index.to_list(),
                                      'model_params(M)': model,
                                      'embed_params (K)': embed,
                                      # 'pos_params': pos,
                                      'all_params(M)': all_added}).sort_values(by='model_params(M)')
    calculated_params.to_csv(os.path.join(data_path, 'calculated_params.csv'), index=False)

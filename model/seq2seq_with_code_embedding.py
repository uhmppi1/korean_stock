from keras.models import Model, Input
from keras.layers import Dense, CuDNNLSTM, Embedding, Concatenate
from model.repeat_layer import RepeatEmbedding

def get_model(hidden_size=50, input_shape=(10, 3), decoder_input_shape=(5,3), code_num=20, embed_size=2, output_dim=1):
    # define model
    input_seq_len = input_shape[0]

    encoder_inputs_1 = Input(shape=input_shape)
    encoder_inputs_2 = Input(shape=(1,))
    decoder_inputs = Input(shape=decoder_input_shape)

    inf_decoder_inputs = Input(shape=(None,decoder_input_shape[1]), name="inf_decoder_inputs")
    state_h_inputs = Input(shape=(hidden_size,), name="state_input_h")
    state_c_inputs = Input(shape=(hidden_size,), name="state_input_c")

    code_emb = Embedding(input_dim=code_num, output_dim=embed_size)
    company_embedding = code_emb(encoder_inputs_2)
    company_embeddings = RepeatEmbedding(input_seq_len)(company_embedding)
    input_concatenated = Concatenate(axis=2)([encoder_inputs_1, company_embeddings])
    encoder_lstm = CuDNNLSTM(hidden_size, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(input_concatenated)
    encoder_states = [state_h, state_c]

    decoder_lstm = CuDNNLSTM(hidden_size, return_sequences=True, return_state=True)
    decoder_lstm_out, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_dense = Dense(output_dim)
    decoder_output = decoder_dense(decoder_lstm_out)

    seq2seq_model = Model([encoder_inputs_1, encoder_inputs_2], decoder_output)
    print('### seq2seq_model ###')
    seq2seq_model.summary()

    encoder_model = Model([encoder_inputs_1, encoder_inputs_2], [encoder_outputs, state_h, state_c])
    print('### encoder_model ###')
    encoder_model.summary()


    inf_decoder_res, decoder_h, decoder_c = decoder_lstm(inf_decoder_inputs, initial_state=[state_h_inputs, state_c_inputs])

    inf_decoder_out = decoder_dense(inf_decoder_res)
    decoder_model = Model(inputs=[inf_decoder_inputs, state_h_inputs, state_c_inputs],
                      outputs=[inf_decoder_out, decoder_h, decoder_c])
    print('### decoder_model ###')
    decoder_model.summary()

    return seq2seq_model, encoder_model, decoder_model



from keras.models import Model, Input
from keras.layers import Dense, CuDNNLSTM, Embedding, Concatenate, TimeDistributed, Bidirectional, Dot, Activation
from model.repeat_layer import RepeatEmbedding

def get_model(hidden_size=50, input_shape=(10, 3), decoder_input_shape=(5,3), code_num=20, embed_size=2, output_dim=3):
    # define model
    input_seq_len = input_shape[0]

    encoder_inputs_1 = Input(shape=input_shape)
    encoder_inputs_2 = Input(shape=(1,))
    decoder_inputs = Input(shape=decoder_input_shape)

    code_emb = Embedding(input_dim=code_num, output_dim=embed_size)
    company_embedding = code_emb(encoder_inputs_2)
    company_embeddings = RepeatEmbedding(input_seq_len)(company_embedding)
    input_concatenated = Concatenate(axis=2)([encoder_inputs_1, company_embeddings])

    encoder_lstm = Bidirectional(CuDNNLSTM(units=hidden_size, return_sequences=True), merge_mode='concat')
    encoder_outputs = encoder_lstm(input_concatenated)

    decoder_lstm = CuDNNLSTM(units=hidden_size * 2, return_sequences=True, return_state=True)
    decoder_lstm_out, _, _ = decoder_lstm(decoder_inputs)

    attention = Dot(axes=[2, 2])([decoder_lstm_out, encoder_outputs])
    attention = Activation('softmax')(attention)
    context = Dot(axes=[2, 1])([attention, encoder_outputs])

    decoder_combined_context = Concatenate(axis=2)([context, decoder_lstm_out])
    output = TimeDistributed(Dense(hidden_size, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dim))(output)

    model = Model([encoder_inputs_1, encoder_inputs_2, decoder_inputs], output)
    model.summary()

    return model
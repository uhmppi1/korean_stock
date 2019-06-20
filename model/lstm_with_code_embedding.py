from keras.models import Model, Input
from keras.layers import Dense, LSTM, Embedding, Concatenate
from model.repeat_layer import RepeatEmbedding

def get_model(hidden_size=50, input_shape=(10, 3), code_num=20, embed_size=2, output_dim=1):
    # define model
    seq_len = input_shape[0]

    encoder_inputs_1 = Input(shape=input_shape)
    encoder_inputs_2 = Input(shape=(1,))

    code_emb = Embedding(input_dim=code_num, output_dim=embed_size)
    company_embedding = code_emb(encoder_inputs_2)
    company_embeddings = RepeatEmbedding(seq_len)(company_embedding)
    input_concatenated = Concatenate(axis=2)([encoder_inputs_1, company_embeddings])
    lstm_output = LSTM(hidden_size, activation='relu')(input_concatenated)
    model_output = Dense(output_dim)(lstm_output)

    model = Model([encoder_inputs_1, encoder_inputs_2], model_output)
    model.summary()

    return model



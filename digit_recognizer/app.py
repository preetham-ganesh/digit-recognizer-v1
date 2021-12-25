import os
from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import pickle
import logging
import sentencepiece as spm
import unicodedata
import re
import tensorflow_datasets as tfds
from import_file import import_file
import spacy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c
    return out

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w, lang):
    w = w.lower().strip()
    if 'http' in w or 'https' in w or 'xml' in w:
        return 0
    w = remove_html_markup(w)
    if '{}' in w:
        return 0
    w = w.replace('\n', ' ')
    w = w.replace('\t', ' ')
    if w == '':
        return 0
    else:
        w = w.replace('##at##-##at##', '-')
        w = w.replace('&apos;', "'")
        w = w.replace('&quot;', '"')
        w = w.replace('&#91;', "")
        w = w.replace('&#93;', "")
        w = w.replace('&#124;', "")
        w = w.replace(' ', ' ')
        if lang == 'en':
            w = unicode_to_ascii(w)
            w = re.sub(r"[^-!$&(),./%0-9:;?a-z€'\"]+", " ", w)
        elif lang == 'es':
            w = re.sub(r"[^-!$&(),./%0-9:;?a-záéíñóúü¿¡€'\"]+", " ", w)
        elif lang == 'de':
            w = re.sub(r"[^-!$&(),./%0-9:;?a-z€äöüß'\"]+", " ", w)
        else:
            w = re.sub(r"[^-!$&(),./%0-9:;?a-zùûüÿ€àâæçéèêëïîôœ«»'\"]+", " ", w)
        w = re.sub('\.{2,}', '.', w)
        w = re.sub(r'(\d)th', r'\1 th', w, flags=re.I)
        w = re.sub(r'(\d)st', r'\1 st', w, flags=re.I)
        w = re.sub(r'(\d)rd', r'\1 rd', w, flags=re.I)
        w = re.sub(r'(\d)nd', r'\1 nd', w, flags=re.I)
        punc = list("-!$&(),./%:;?€'¿¡«»")
        for i in punc:
            w = w.replace(i, " "+i+" ")
        w = w.replace('"', ' " ')
        w = w.strip()
        w = re.sub(r'\s+', ' ', w)
        return w

def open_file(name):
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/'
    with open(loc_to + name + '.pkl', 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def pos_tagging(inp_sent, inp_lang):
    if inp_lang == 'en':
        nlp = spacy.load(inp_lang+'_core_web_sm')
    else:
        nlp = spacy.load(inp_lang + '_core_news_sm')
    result = nlp(inp_sent)
    new_inp_sent = []
    for i in result:
        print(i.text, i.pos_)
        if i.pos_ == 'PROPN':
            new_inp_sent.append('<'+'#'.join(i.text)+'>')
        else:
            new_inp_sent.append(i.text)
    return ' '.join(new_inp_sent)

def model_2_translate(inp_sent, inp_lang, tar_lang):
    model = import_file('/home/preetham/Documents/neural-machine-translation/models/codes/' + inp_lang + '-' +
                        tar_lang + '/model_2/model.py')
    from model import Transformer
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/'
    inp_sent = preprocess_sentence(inp_sent, inp_lang)
    inp_sent = pos_tagging(inp_sent, inp_lang)
    parameters = open_file(inp_lang + '-' + tar_lang + '/model_2/utils/parameters')
    inp_token = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_to + inp_lang + '-' + tar_lang +
                                                                       '/model_2/tokenizer/' + inp_lang + '-tokenizer')
    tar_token = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_to + inp_lang + '-' + tar_lang +
                                                                       '/model_2/tokenizer/' + tar_lang + '-tokenizer')
    inp_sent = inp_token.encode(inp_sent)
    inp_sent = [inp_token.vocab_size] + inp_sent + [inp_token.vocab_size + 1]
    inp_sent = tf.expand_dims(inp_sent, 0)
    global transformer
    transformer = Transformer(parameters['n_layers'], parameters['d_model'], parameters['n_heads'], parameters['dff'],
                              parameters['inp_vocab_size'], parameters['tar_vocab_size'],
                              pe_input=parameters['inp_vocab_size'], pe_target=parameters['tar_vocab_size'],
                              rate=parameters['dropout'])
    checkpoint = tf.train.Checkpoint(transformer=transformer)
    checkpoint_dir = loc_to + inp_lang+'-'+tar_lang+'/model_2/training_checkpoints'
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    dec_inp = tf.expand_dims([tar_token.vocab_size], 0)
    for i in range(1, parameters['tar_max_length']):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_sent, dec_inp)
        predictions = transformer(inp_sent, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tar_token.vocab_size + 1 == predicted_id:
            dec_inp = tf.concat([dec_inp, predicted_id], axis=-1)
            break
        dec_inp = tf.concat([dec_inp, predicted_id], axis=-1)
    dec_inp = tf.squeeze(dec_inp, axis=0)
    tar_sent = tar_token.decode([i for i in dec_inp.numpy()[1:-1]])
    return tar_sent

def model_1_translate(inp_sent, inp_lang, tar_lang):
    model = import_file('/home/preetham/Documents/neural-machine-translation/models/codes/' + inp_lang + '-' +
                        tar_lang + '/model_1/model.py')
    from model import Encoder, Decoder
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/'
    inp_sent = preprocess_sentence(inp_sent, inp_lang)
    inp_sent = pos_tagging(inp_sent, inp_lang)
    inp_word_index = open_file(inp_lang+'-'+tar_lang+'/model_1/utils/inp-word-index')
    tar_index_word = open_file(inp_lang+'-'+tar_lang+'/model_1/utils/tar-index-word')
    tar_word_index = open_file(inp_lang+'-'+tar_lang+'/model_1/utils/tar-word-index')
    parameters = open_file(inp_lang+'-'+tar_lang+'/model_1/utils/parameters')
    sp = spm.SentencePieceProcessor()
    sp.Load(loc_to + inp_lang + '-' + tar_lang + '/model_1/tokenizer/' + inp_lang + '.model')
    inp_sent = sp.EncodeAsPieces(inp_sent)
    inp_sent = [[inp_word_index[i] for i in inp_sent]]
    inp_sent = tf.keras.preprocessing.sequence.pad_sequences(inp_sent, maxlen=parameters['inp_max_length'],
                                                             padding='post')
    inp_sent = tf.convert_to_tensor(inp_sent)
    tar_sent = []
    encoder = Encoder(parameters['emb_size'], parameters['inp_vocab_size'], parameters['rnn_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    checkpoint_dir = loc_to + inp_lang+'-'+tar_lang+'/model_1/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    hidden = encoder.initialize_hidden_state(1, parameters['rnn_size'])
    enc_out, enc_hidden = encoder(inp_sent, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([tar_word_index['<s>']], 0)
    for i in range(1, parameters['tar_max_length']):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        predicted_id = tf.argmax(prediction[0]).numpy()
        if tar_index_word[predicted_id] != '</s>':
            tar_sent.append(tar_index_word[predicted_id])
        else:
            break
        dec_inp = tf.expand_dims([predicted_id], 0)
    sp.Load(loc_to + inp_lang + '-' + tar_lang + '/model_1/tokenizer/' + inp_lang + '.model')
    tar_sent = sp.DecodePieces(tar_sent)
    tar_sent = tar_sent.replace('▁', ' ')
    return tar_sent

def tar_sent_post_process(tar_sent):
    tar_sent = tar_sent.replace('<', '')
    tar_sent = tar_sent.replace('#', '')
    tar_sent = tar_sent.replace('>', '')
    return tar_sent

@app.route("/translate_sentences", methods=["POST"])
def translate_sentences():
    inp_lang = request.form.getlist('inp_lang')[0]
    tar_lang = request.form.getlist('tar_lang')[0]
    model_name = request.form.getlist('model_name')[0]
    inp_sent = request.form.getlist('inp_sent')[0]
    tar_sent = ''
    print()
    print('Input Language: ', inp_lang)
    print('Target Language: ', tar_lang)
    print('Model name: ', model_name)
    print('Input Sentence: ', inp_sent)
    print()
    if inp_lang == tar_lang:
        tar_sent = inp_sent
    elif inp_lang == 'en' or tar_lang == 'en':
        if model_name == 'model_1':
            tar_sent = model_1_translate(inp_sent, inp_lang, tar_lang)
            tar_sent = tar_sent_post_process(tar_sent)
        else:
            tar_sent = model_2_translate(inp_sent, inp_lang, tar_lang)
            tar_sent = tar_sent_post_process(tar_sent)
    else:
        if model_name == 'model_1':
            tar_sent = model_1_translate(inp_sent, inp_lang, 'en')
            tar_sent = model_1_translate(tar_sent, 'en', tar_lang)
            tar_sent = tar_sent_post_process(tar_sent)
        else:
            tar_sent = model_2_translate(inp_sent, inp_lang, 'en')
            tar_sent = model_2_translate(tar_sent, 'en', tar_lang)
            tar_sent = tar_sent_post_process(tar_sent)
    return render_template("result.html", inp_lang=inp_lang, tar_lang=tar_lang, model_name=model_name,
                           inp_sent=inp_sent, tar_sent=tar_sent)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

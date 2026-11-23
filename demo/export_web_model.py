"""Export the trained custom Transformer for the browser demo in docs/.

Reads the Keras weights and tokenizers from Codes and Models/CustomTransformer and writes:
  docs/model/encoder.onnx, docs/model/decoder.onnx   (weights stored as float16)
  docs/model/vocab.json                              (tokenizer vocabularies)
  docs/model/manifest.json                           (file sizes for the download progress bar)

The number-to-word table used by docs/preprocess.js comes from indic-num2words 1.2.2, the version
that matches the training vocabulary (newer versions spell some numbers differently).

Usage (from the repository root):
  pip install h5py numpy onnx indic-num2words==1.2.2
  python demo/export_web_model.py
"""
import io
import json
import os
import pickle
import zipfile

import h5py
import numpy as np
import onnx
from onnx import helper as h, TensorProto as TP, numpy_helper as nh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'Codes and Models', 'CustomTransformer')
OUT = os.path.join(ROOT, 'docs', 'model')

D_MODEL, HEADS, LAYERS, MAX_IN, MAX_OUT = 300, 10, 4, 300, 75
DEPTH = D_MODEL // HEADS


# Tokenizers ------------------------------------------------------------------
class _Tokenizer:
    def __setstate__(self, state):
        self.__dict__.update(state)


class _SafeUnpickler(pickle.Unpickler):
    """Loads the pickled Keras Tokenizer as plain data, without importing Keras."""
    def find_class(self, module, name):
        if module.startswith('keras') and name == 'Tokenizer':
            return _Tokenizer
        if module == 'collections' and name in ('OrderedDict', 'defaultdict'):
            import collections
            return getattr(collections, name)
        if module == 'builtins' and name in ('int', 'float', 'str', 'dict', 'list', 'set', 'tuple'):
            import builtins
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f'unexpected object in tokenizer pickle: {module}.{name}')


def load_tokenizer(name):
    with open(os.path.join(SRC, name), 'rb') as f:
        return _SafeUnpickler(f).load()


def export_vocab():
    art, summ = load_tokenizer('article_tokenizer.pickle'), load_tokenizer('summary_tokenizer.pickle')
    summary_words = [None] * summ.num_words
    for i, w in summ.index_word.items():
        if int(i) < summ.num_words:
            summary_words[int(i)] = w
    vocab = {
        'article': {w: i for w, i in art.word_index.items() if i < art.num_words},
        'summary': summary_words,
        'start_in': art.word_index['starttoken'], 'end_in': art.word_index['endtoken'],
        'start_out': summ.word_index['starttoken'], 'end_out': summ.word_index['endtoken'],
        'max_in': MAX_IN, 'max_out': MAX_OUT,
    }
    with open(os.path.join(OUT, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(',', ':'))


# ONNX graphs -----------------------------------------------------------------
def pos_encoding(n, d):
    pos = np.arange(n)[:, None]
    dims = np.arange(d)[None, :]
    ang = pos / np.power(10000, (2 * (dims // 2)) / np.float32(d))
    ang[:, 0::2] = np.sin(ang[:, 0::2])
    ang[:, 1::2] = np.cos(ang[:, 1::2])
    return ang.astype(np.float32)[None]


class Graph:
    def __init__(self, weights):
        self.w, self.nodes, self.inits, self.n, self.loaded = weights, [], [], 0, {}

    def name(self, p):
        self.n += 1
        return f'{p}_{self.n}'

    def const(self, arr, p='c'):
        nm = self.name(p)
        self.inits.append(nh.from_array(np.asarray(arr), nm))
        return nm

    def weight(self, path):
        """Model weights are stored as float16 and cast back to float32 when the model loads."""
        if path not in self.loaded:
            safe = path.replace('/', '.') + '.f16'
            self.inits.append(nh.from_array(self.w[path].astype(np.float16), safe))
            self.loaded[path] = self.op('Cast', [safe], to=TP.FLOAT)
        return self.loaded[path]

    def op(self, typ, ins, **attrs):
        out = self.name(typ.lower())
        self.nodes.append(h.make_node(typ, ins, [out], **attrs))
        return out

    def dense(self, x, p):
        return self.op('Add', [self.op('MatMul', [x, self.weight(p + '/vars/0')]), self.weight(p + '/vars/1')])

    def layer_norm(self, x, p):
        return self.op('LayerNormalization', [x, self.weight(p + '/vars/0'), self.weight(p + '/vars/1')],
                       axis=-1, epsilon=1e-6)

    def attention(self, v, k, q, bias, p):
        heads = self.const(np.array([1, -1, HEADS, DEPTH], np.int64), 'shape')
        split = lambda x, perm: self.op('Transpose', [self.op('Reshape', [x, heads])], perm=perm)
        q = split(self.dense(q, p + '/wq'), [0, 2, 1, 3])
        k = split(self.dense(k, p + '/wk'), [0, 2, 3, 1])
        v = split(self.dense(v, p + '/wv'), [0, 2, 1, 3])
        logits = self.op('Mul', [self.op('MatMul', [q, k]), self.const(np.float32(1 / np.sqrt(np.float32(DEPTH))), 'scale')])
        attn = self.op('Softmax', [self.op('Add', [logits, bias])], axis=-1)
        o = self.op('Transpose', [self.op('MatMul', [attn, v])], perm=[0, 2, 1, 3])
        o = self.op('Reshape', [o, self.const(np.array([1, -1, D_MODEL], np.int64), 'shape')])
        return self.dense(o, p + '/dense')

    def embed(self, ids, table, pe):
        x = self.op('Gather', [self.weight(table), ids], axis=0)
        x = self.op('Mul', [x, self.const(np.float32(np.sqrt(np.float32(D_MODEL))), 'sqrtd')])
        return self.op('Add', [x, pe])

    def padding_bias(self, ids, shape):
        m = self.op('Cast', [self.op('Equal', [ids, self.const(np.array(0, np.int64), 'zero')])], to=TP.FLOAT)
        return self.op('Mul', [self.op('Reshape', [m, shape]), self.const(np.float32(-1e9), 'neg')])


def build_encoder(w):
    g = Graph(w)
    x = g.embed('ids', 'encoder/embedding/vars/0', g.const(pos_encoding(MAX_IN, D_MODEL), 'pe'))
    bias = g.padding_bias('ids', g.const(np.array([1, 1, 1, MAX_IN], np.int64), 'shape'))
    for i in range(LAYERS):
        p = 'encoder/enc_layers/encoder_layer' + ('' if i == 0 else f'_{i}')
        x = g.layer_norm(g.op('Add', [x, g.attention(x, x, x, bias, p + '/mha')]), p + '/layernorm1')
        f = g.dense(g.op('Relu', [g.dense(x, p + '/ffn/layers/dense')]), p + '/ffn/layers/dense_1')
        x = g.layer_norm(g.op('Add', [x, f]), p + '/layernorm2')
    g.nodes.append(h.make_node('Identity', [x], ['enc']))
    return h.make_graph(g.nodes, 'encoder', [h.make_tensor_value_info('ids', TP.INT64, [1, MAX_IN])],
                        [h.make_tensor_value_info('enc', TP.FLOAT, [1, MAX_IN, D_MODEL])], g.inits)


def build_decoder(w):
    g = Graph(w)
    t = g.op('Slice', [g.op('Shape', ['tar']), g.const(np.array([1], np.int64), 's'), g.const(np.array([2], np.int64), 'e')])
    pe = g.op('Slice', [g.const(pos_encoding(MAX_IN, D_MODEL), 'pe'), g.const(np.array([0], np.int64), 's'), t,
                        g.const(np.array([1], np.int64), 'axis')])
    x = g.embed('tar', 'decoder/embedding/vars/0', pe)

    # Self-attention mask: a word may only look at earlier words (and never at padding)
    ones = g.op('ConstantOfShape', [g.op('Concat', [t, t], axis=0)], value=nh.from_array(np.array([1.0], np.float32)))
    look_ahead = g.op('Sub', [g.const(np.float32(1.0), 'one'), g.op('Trilu', [ones], upper=0)])
    pad = g.op('Cast', [g.op('Equal', ['tar', g.const(np.array(0, np.int64), 'zero')])], to=TP.FLOAT)
    pad = g.op('Reshape', [pad, g.op('Concat', [g.const(np.array([1, 1, 1], np.int64), 'c'), t], axis=0)])
    self_bias = g.op('Mul', [g.op('Max', [pad, look_ahead]), g.const(np.float32(-1e9), 'neg')])
    enc_bias = g.padding_bias('ids', g.const(np.array([1, 1, 1, MAX_IN], np.int64), 'shape'))

    for i in range(LAYERS):
        p = 'decoder/dec_layers/decoder_layer' + ('' if i == 0 else f'_{i}')
        o1 = g.layer_norm(g.op('Add', [g.attention(x, x, x, self_bias, p + '/mha1'), x]), p + '/layernorm1')
        o2 = g.layer_norm(g.op('Add', [g.attention('enc', 'enc', o1, enc_bias, p + '/mha2'), o1]), p + '/layernorm2')
        f = g.dense(g.op('Relu', [g.dense(o2, p + '/ffn/layers/dense')]), p + '/ffn/layers/dense_1')
        x = g.layer_norm(g.op('Add', [f, o2]), p + '/layernorm3')
    last = g.op('Gather', [x, g.const(np.array(-1, np.int64), 'last')], axis=1)
    g.nodes.append(h.make_node('Identity', [g.dense(last, 'final_layer')], ['logits']))
    return h.make_graph(g.nodes, 'decoder',
                        [h.make_tensor_value_info('tar', TP.INT64, [1, 'T']),
                         h.make_tensor_value_info('enc', TP.FLOAT, [1, MAX_IN, D_MODEL]),
                         h.make_tensor_value_info('ids', TP.INT64, [1, MAX_IN])],
                        [h.make_tensor_value_info('logits', TP.FLOAT, [1, 8461])], g.inits)


def export_onnx():
    with zipfile.ZipFile(os.path.join(SRC, 'my_model_transformer.keras')) as z:
        data = z.read('model.weights.h5')
    weights = {}
    with h5py.File(io.BytesIO(data), 'r') as f:
        f.visititems(lambda n, o: weights.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    sizes = {}
    for name, build in [('encoder', build_encoder), ('decoder', build_decoder)]:
        model = h.make_model(build(weights), opset_imports=[h.make_opsetid('', 17)], producer_name='hindi-summarizer')
        model.ir_version = 8
        onnx.checker.check_model(model)
        path = os.path.join(OUT, f'{name}.onnx')
        onnx.save(model, path)
        sizes[name] = os.path.getsize(path)
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(sizes, f)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    export_vocab()
    export_onnx()
    print('Wrote', ', '.join(sorted(os.listdir(OUT))))

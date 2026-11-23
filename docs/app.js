import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { preprocess, words, encodeArticle, decodeSummary } from './preprocess.js';

ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;   // GitHub Pages can't enable the headers multi-threading needs

const $ = (id) => document.getElementById(id);
const els = {
  article: $('article'), run: $('run'), status: $('status'), count: $('word-count'),
  progress: $('progress'), bar: $('progress-bar'), examples: $('examples'),
  empty: $('empty'), summary: $('summary'), summaryText: $('summary-text'), meta: $('meta'),
  translate: $('translate'), reference: $('reference'), referenceText: $('reference-text'),
  read: $('read'), tokens: $('tokens'),
};

let vocab = null, examples = [], manifest = null, sessions = null, busy = false, activeExample = null;

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.classList.toggle('error', isError);
}

function updateCount() {
  const n = els.article.value.trim() ? els.article.value.trim().split(/\s+/).length : 0;
  els.count.textContent = n ? `${n.toLocaleString('en-US')} words` : '';
}

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download ${url} (HTTP ${res.status})`);
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onBytes(value.length);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

async function loadModel() {
  if (sessions) return sessions;
  const total = manifest.encoder + manifest.decoder;
  const mb = (b) => Math.round(b / 1e6);
  let done = 0;
  els.progress.hidden = false;
  const tick = (n) => {
    done += n;
    els.bar.style.width = `${Math.min(100, (done / total) * 100)}%`;
    setStatus(`Downloading the model (${mb(done)} of ${mb(total)} MB). Your browser keeps it for next time.`);
  };
  tick(0);
  const [encBytes, decBytes] = await Promise.all([
    fetchWithProgress('model/encoder.onnx', tick),
    fetchWithProgress('model/decoder.onnx', tick),
  ]);
  setStatus('Loading the model');
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  sessions = {
    encoder: await ort.InferenceSession.create(encBytes, opts),
    decoder: await ort.InferenceSession.create(decBytes, opts),
  };
  els.progress.hidden = true;
  return sessions;
}

const int64 = (arr, dims) => new ort.Tensor('int64', BigInt64Array.from(arr, (v) => BigInt(v)), dims);
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

function renderTokens(clean) {
  const frag = document.createDocumentFragment();
  let known = 0, cutting = false;
  const limit = vocab.max_in - 1;          // the start token takes the first slot
  for (const w of words(clean)) {
    const isKnown = vocab.article[w] !== undefined;
    if (isKnown && known === limit && !cutting) {
      cutting = true;
      const mark = document.createElement('span');
      mark.className = 'limit';
      mark.textContent = '300-word limit. The model ignores everything below this line.';
      frag.append(mark);
    }
    if (isKnown) known++;
    const span = document.createElement('span');
    span.textContent = w;
    if (!isKnown) span.className = 'unk';
    if (cutting) span.classList.add('cut');
    frag.append(span, ' ');
  }
  els.tokens.replaceChildren(frag);
  els.tokens.scrollTop = 0;
}

function showReference() {
  const ex = activeExample;
  const matches = ex && els.article.value.trim() === ex.article;
  els.reference.hidden = !matches;
  if (matches) els.referenceText.textContent = ex.reference;
}

async function summarize() {
  if (busy) return;
  const raw = els.article.value.trim();
  if (!raw) { setStatus('Paste a Hindi news article first, or pick an example.', true); return; }

  const clean = preprocess(raw);
  const enc = encodeArticle(clean, vocab);
  if (enc.known < 5) {
    setStatus('The model recognized almost none of these words. It only reads Hindi news articles.', true);
    return;
  }

  busy = true;
  els.run.disabled = true;
  try {
    const { encoder, decoder } = await loadModel();
    setStatus('Writing the summary');
    const t0 = performance.now();

    els.empty.hidden = true;
    els.summary.hidden = false;
    els.summary.classList.add('writing');
    els.summaryText.textContent = '';
    els.meta.hidden = true;
    els.translate.hidden = true;
    els.reference.hidden = true;
    renderTokens(clean);
    els.read.hidden = false;
    if (window.matchMedia('(max-width: 60rem)').matches) {
      const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      $('summary-heading').scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
    }

    const ids = int64(enc.ids, [1, vocab.max_in]);
    const encOut = (await encoder.run({ ids })).enc;
    const out = [vocab.start_out];
    for (let step = 0; step < vocab.max_out; step++) {
      const res = await decoder.run({ tar: int64(out, [1, out.length]), enc: encOut, ids });
      const next = argmax(res.logits.data);
      if (next === vocab.end_out) break;
      out.push(next);
      els.summaryText.textContent = decodeSummary(out.slice(1), vocab);
      await nextFrame();
    }

    const text = decodeSummary(out.slice(1), vocab);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    els.summary.classList.remove('writing');
    els.summaryText.textContent = text || '(The model returned an empty summary.)';
    const fmt = (n) => n.toLocaleString('en-US');
    els.meta.textContent = `After preprocessing, the model recognized ${fmt(enc.known)} of ${fmt(enc.words)} words`
      + (enc.truncated ? ', used the first 300' : '')
      + ` and took ${secs} seconds.`;
    els.meta.hidden = false;
    if (text) {
      els.translate.href = `https://translate.google.com/?sl=hi&tl=en&op=translate&text=${encodeURIComponent(text)}`;
      els.translate.hidden = false;
    }
    showReference();
    setStatus('');
  } catch (err) {
    console.error(err);
    els.summary.classList.remove('writing');
    els.progress.hidden = true;
    setStatus(`The model failed to run: ${err.message}. Reload the page and try again.`, true);
  } finally {
    busy = false;
    els.run.disabled = false;
  }
}

function renderExamples() {
  for (const ex of examples) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = ex.label;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => {
      if (busy) return;
      activeExample = ex;
      for (const c of els.examples.children) c.setAttribute('aria-pressed', String(c === b));
      els.article.value = ex.article;
      updateCount();
      summarize();
    });
    els.examples.append(b);
  }
}

async function init() {
  try {
    const [v, e, m] = await Promise.all(
      ['model/vocab.json', 'model/examples.json', 'model/manifest.json'].map((u) => fetch(u).then((r) => {
        if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
        return r.json();
      })));
    vocab = v; examples = e; manifest = m;
  } catch (err) {
    setStatus(`The page could not load its data files (${err.message}). Reload to try again.`, true);
    els.run.disabled = true;
    return;
  }
  renderExamples();
  els.run.addEventListener('click', summarize);
  els.article.addEventListener('input', () => {
    updateCount();
    if (activeExample && els.article.value.trim() !== activeExample.article) {
      activeExample = null;
      for (const c of els.examples.children) c.setAttribute('aria-pressed', 'false');
    }
  });
  els.article.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) summarize();
  });
}

init();

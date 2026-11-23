// Text preprocessing and tokenization, ported from the training notebook
// (preprocess_text + Keras Tokenizer). Matches the Python version token for token.

const HI = {"0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार", "5": "पाँच", "6": "छः", "7": "सात", "8": "आठ", "9": "नौ", "10": "दस", "11": "ग्यारह", "12": "बारह", "13": "तेरह", "14": "चौदह", "15": "पंद्रह", "16": "सोलह", "17": "सत्रह", "18": "अट्ठारह", "19": "उन्नीस", "20": "बीस", "21": "इक्कीस", "22": "बाईस", "23": "तेईस", "24": "चौबिस", "25": "पच्चीस", "26": "छब्बीस", "27": "सत्ताईस", "28": "अट्ठाईस", "29": "उनतीस", "30": "तीस", "31": "इकतीस", "32": "बत्तीस", "33": "तैंतीस", "34": "चौंतीस", "35": "पैंतीस", "36": "छत्तीस", "37": "सैंतीस", "38": "अड़तीस", "39": "उनतालीस", "40": "चालीस", "41": "इकतालीस", "42": "बयालीस", "43": "तैंतालीस", "44": "चौंतालीस", "45": "पैंतालीस", "46": "छियालीस", "47": "सैंतालीस", "48": "अड़तालीस", "49": "उनचास", "50": "पचास", "51": "इक्यावन", "52": "बावन", "53": "तिरेपन", "54": "चौवन", "55": "पचपन", "56": "छप्पन", "57": "सत्तावन", "58": "अट्ठावन", "59": "उनसठ", "60": "साठ", "61": "इकसठ", "62": "बासठ", "63": "तिरेसठ", "64": "चौंसठ", "65": "पैंसठ", "66": "छयासठ", "67": "सरसठ", "68": "अड़सठ", "69": "उनहत्तर", "70": "सत्तर", "71": "इकहत्तर", "72": "बहत्तर", "73": "तिहत्तर", "74": "चौहत्तर", "75": "पचहत्तर", "76": "छिहत्तर", "77": "सतहत्तर", "78": "अठहत्तर", "79": "उन्यासी", "80": "अस्सी", "81": "इक्यासी", "82": "बयासी", "83": "तिरासी", "84": "चौरासी", "85": "पचासी", "86": "छियासी", "87": "सत्तासी", "88": "अठासी", "89": "नवासी", "90": "नब्बे", "91": "इक्यानवे", "92": "बानवे", "93": "तिरानवे", "94": "चौरानवे", "95": "पचानवे", "96": "छियानवे", "97": "सत्तानवे", "98": "अट्ठानवे", "99": "निन्यानवे", "100": "सौ", "1000": "हज़ार", "100000": "लाख", "10000000": "करोड़", "1000000000": "अरब"};

// Python's \s for str patterns (str.isspace characters)
const WS = '\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const PUNCT = new RegExp(`[\\n\\r\\t_\\-~\\+\\.${WS}]{2,}|[<>()|&©@#ø\\[\\]'",;:.।‘’]`, 'gu');
const MULTI_WS = new RegExp(`[${WS}]+`, 'gu');
const EMOJI = /(?:\p{Extended_Pictographic}|\p{Regional_Indicator})(?:️|⃣|[\u{1F3FB}-\u{1F3FF}]|‍(?:\p{Extended_Pictographic}|\p{Regional_Indicator})|\p{Regional_Indicator})*/gu;
const IS_WORD = /[\p{L}\p{N}_]/u;           // Python's \w
const DIGIT_RUN = /\p{Nd}+/gu;               // Python's \d+

function digitValue(ch) {
  const c = ch.codePointAt(0);
  if (c >= 0x30 && c <= 0x39) return c - 0x30;
  if (c >= 0x966 && c <= 0x96f) return c - 0x966;          // Devanagari
  // Other scripts: decimal digits sit in runs of ten that start at a code point
  // whose value is zero, so step back until the previous code point is not a digit.
  let z = c;
  while (z > 0 && /\p{Nd}/u.test(String.fromCodePoint(z - 1)) && c - z < 9) z--;
  return c - z;
}

// num_to_word(num, lang='hi') from indic-num2words 1.2.x
function numToWord(digits) {
  const sep = ', ', comb = ' ';
  let num = digits.replace(/^0+/, '');
  const allTwoDigit = (d) => {
    if (d.length <= 1) return HI[d] ?? '';
    if (d === '00') return HI['0'] + sep + HI['0'];
    if (d[0] === '0') return HI['0'] + sep + HI[d[1]];
    return HI[d];
  };
  const twoDigit = (d) => { d = d.replace(/^0+/, ''); return d.length ? HI[d] : ''; };
  const allDigit = (d) => {
    d = d.replace(/^0+/, '');
    const n = d.length;
    if (n > 3) {
      const take = (n % 2) + 1;
      const base = '1' + '0'.repeat(Math.trunc(n / 2) * 2 - 1);
      return HI[d.slice(0, take)] + comb + HI[base] + sep + allDigit(d.slice(take));
    } else if (n === 3) {
      return HI[d.slice(0, 1)] + comb + HI['100'] + sep + twoDigit(d.slice(1));
    }
    return twoDigit(d);
  };
  let out;
  if (num.length === 0) out = HI['0'];
  else if (num.length <= 9) out = allDigit(num);
  else {
    const it = num.length % 2 === 0 ? num.length / 2 : pyRoundHalfEven(num.length / 2);
    out = allTwoDigit(num.slice(0, 2));
    for (let i = 1; i < it; i++) out = out + sep + allTwoDigit(num.slice(i * 2, (i + 1) * 2));
    const rest = num.slice(it * 2);
    if (allTwoDigit(rest) !== '') out = out + sep + allTwoDigit(rest);
  }
  return stripChars(out, sep);
}

function pyRoundHalfEven(x) {            // Python's round() for .5 cases
  const f = Math.floor(x);
  return f % 2 === 0 ? f : f + 1;
}

function stripChars(s, chars) {          // Python's str.strip(chars)
  let a = 0, b = s.length;
  while (a < b && chars.includes(s[a])) a++;
  while (b > a && chars.includes(s[b - 1])) b--;
  return s.slice(a, b);
}

function convertNumbers(text) {
  return text.replace(DIGIT_RUN, (run, offset, whole) => {
    const prevCh = offset > 0 ? [...whole.slice(Math.max(0, offset - 2), offset)].pop() : ' ';
    const nextCh = [...whole.slice(offset + run.length)][0] ?? ' ';
    if (IS_WORD.test(prevCh) || IS_WORD.test(nextCh)) return run;
    const ascii = [...run].map(digitValue).join('');
    return numToWord(ascii);
  });
}

export function preprocess(raw) {
  let t = String(raw).replace(EMOJI, ' ');
  t = t.replaceAll('‍', '').toLowerCase().replace(PUNCT, ' ');
  t = t.replace(/[a-zA-Z]/g, ' ').toLowerCase();
  t = t.replace(MULTI_WS, ' ').toLowerCase();
  t = convertNumbers(t);
  t = t.replaceAll('‍', '').toLowerCase().replace(PUNCT, ' ');
  return stripChars(t, ' \t\n\x0b\x0c\r\x1c\x1d\x1e\x1f\x85\xa0                　');
}

const FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';

// Keras Tokenizer.texts_to_sequences (num_words cut-off already applied to the vocab)
// Keras text_to_word_sequence: lowercase, turn filter characters into spaces, split on spaces
export function words(text) {
  let out = '';
  for (const ch of text.toLowerCase()) out += FILTERS.includes(ch) ? ' ' : ch;
  return out.split(' ').filter(Boolean);
}

export function toIds(text, wordIndex) {
  const ids = [];
  for (const w of words(text)) {
    const i = wordIndex[w];
    if (i !== undefined) ids.push(i);
  }
  return ids;
}

export function encodeArticle(clean, vocab) {
  const all = toIds('starttoken ' + clean + ' endtoken', vocab.article);
  const ids = all.slice(0, vocab.max_in);
  while (ids.length < vocab.max_in) ids.push(0);
  return { ids, words: words(clean).length, known: all.length - 2, truncated: all.length > vocab.max_in };
}

export function decodeSummary(ids, vocab) {
  return ids.map((i) => vocab.summary[i]).filter((w) => w != null).join(' ');
}

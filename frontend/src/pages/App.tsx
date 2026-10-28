import { useState, useEffect, useRef } from 'react';

type Cite = {
  document_id: string;
  title: string;
  source_url?: string | null;
  verification_status: string;
  independently_verified?: boolean;
  legal_provisions: string[];
  excerpt: string;
};

type Msg = {
  role: 'user' | 'assistant';
  text: string;
  citations?: Cite[];
};

type Details = Record<string, string>;

function getApiUrl(): string {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined') {
    // If running under local Vite dev server (port 5173), route to local backend on port 8000
    if (window.location.port === '5173') {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
    // In production (Render, Hugging Face, or mobile browser accessing the deployed app),
    // the frontend and backend are hosted on the same origin (no port 8000!)
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
}

const API = getApiUrl();

const examples = [
  'An online seller delivered a damaged phone and is refusing a refund.',
  'My insurer has not processed my claim despite repeated follow-ups.',
  'My flight was cancelled and the airline has not refunded me.',
];

const fields: [keyof Details, string, string][] = [
  ['consumer_name', 'Your name', 'Name as you want it in the letter'],
  ['contact_details', 'Contact details', 'Optional phone or email'],
  ['recipient', 'Recipient', 'Company or grievance officer, if known'],
  ['company', 'Company or seller', 'Who you are writing to'],
  ['product_or_service', 'Product or service', 'What you bought or used'],
  ['transaction_date', 'Transaction date', 'Date, if known'],
  ['amount', 'Amount paid', 'Amount and currency, if known'],
  ['chronology', 'What happened', 'Briefly describe the events'],
  ['remedy_requested', 'What resolution do you want?', 'For example, repair, refund, replacement'],
  ['evidence', 'Documents you can attach', 'Invoice, screenshots, warranty, messages'],
];

function cleanTextForSpeech(text: string): string {
  if (!text) return '';
  return text
    // Replace Indian Rupee symbol with 'rupees'
    .replace(/₹\s*([0-9,]+)/g, '$1 rupees')
    .replace(/₹/g, ' rupees ')
    // Replace links [text](url) with just text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Remove citation references like [S1], [S2, S3]
    .replace(/\[S\d+(?:,\s*S\d+)*\]/gi, '')
    // Remove standalone URLs
    .replace(/https?:\/\/\S+/gi, '')
    // Remove markdown headers #, ##, ###
    .replace(/^[ \t]*#+[ \t]*/gm, '')
    // Remove bold and italics
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    // Remove horizontal rules
    .replace(/^[ \t]*[-*_]{3,}[ \t]*$/gm, '')
    // Remove bullet points
    .replace(/^[ \t]*[-*+]\s+/gm, '')
    // Remove blockquotes
    .replace(/^[ \t]*>[ \t]*/gm, '')
    // Remove inline backticks
    .replace(/`([^`]+)`/g, '$1')
    // Smooth out line breaks into pauses
    .replace(/\n{2,}/g, '. ')
    .replace(/\n/g, ' ')
    // Collapse multiple spaces
    .replace(/\s+/g, ' ')
    .trim();
}

function splitTextIntoChunks(text: string, maxLen = 180): string[] {
  const sentences = text.match(/[^.!?\n]+[.!?\n]+/g) || [text];
  const chunks: string[] = [];
  let current = '';

  for (const s of sentences) {
    const trimmed = s.trim();
    if (!trimmed) continue;
    if ((current + ' ' + trimmed).length <= maxLen) {
      current = current ? current + ' ' + trimmed : trimmed;
    } else {
      if (current) chunks.push(current);
      if (trimmed.length > maxLen) {
        const words = trimmed.split(' ');
        let sub = '';
        for (const w of words) {
          if ((sub + ' ' + w).length <= maxLen) {
            sub = sub ? sub + ' ' + w : w;
          } else {
            if (sub) chunks.push(sub);
            sub = w;
          }
        }
        current = sub;
      } else {
        current = trimmed;
      }
    }
  }
  if (current) chunks.push(current);
  return chunks.length ? chunks : [text];
}

function getPreferredVoice(): SpeechSynthesisVoice | null {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null;

  return (
    voices.find(v => v.lang.toLowerCase() === 'en-in') ||
    voices.find(v => v.lang.toLowerCase().startsWith('en') && (v.name.includes('Natural') || v.name.includes('Neural') || v.name.includes('Online'))) ||
    voices.find(v => v.lang.toLowerCase().startsWith('en-us') || v.lang.toLowerCase().startsWith('en-gb')) ||
    voices.find(v => v.lang.toLowerCase().startsWith('en')) ||
    null
  );
}

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [showDraft, setShowDraft] = useState(false);
  const [draft, setDraft] = useState('');
  const [details, setDetails] = useState<Details>({});

  // Text-to-speech state
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const activeIndexRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.getVoices();
      const onVoices = () => {
        window.speechSynthesis.getVoices();
      };
      window.speechSynthesis.onvoiceschanged = onVoices;
      return () => {
        window.speechSynthesis.cancel();
        window.speechSynthesis.onvoiceschanged = null;
      };
    }
  }, []);

  function stopSpeech() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setSpeakingIndex(null);
    setIsPaused(false);
    activeIndexRef.current = null;
  }

  function speak(index: number, rawText: string) {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported in this browser.');
      return;
    }

    // Toggle pause / resume if the same message is already speaking
    if (speakingIndex === index) {
      if (isPaused) {
        window.speechSynthesis.resume();
        setIsPaused(false);
      } else {
        window.speechSynthesis.pause();
        setIsPaused(true);
      }
      return;
    }

    // Stop existing speech and start new
    window.speechSynthesis.cancel();

    const cleaned = cleanTextForSpeech(rawText);
    if (!cleaned) return;

    const chunks = splitTextIntoChunks(cleaned);
    const voice = getPreferredVoice();

    setSpeakingIndex(index);
    setIsPaused(false);
    activeIndexRef.current = index;

    chunks.forEach((chunk, chunkIdx) => {
      const utt = new SpeechSynthesisUtterance(chunk);
      utt.rate = 1.0;
      utt.pitch = 1.0;
      if (voice) utt.voice = voice;

      if (chunkIdx === chunks.length - 1) {
        utt.onend = () => {
          if (activeIndexRef.current === index) {
            setSpeakingIndex(null);
            setIsPaused(false);
            activeIndexRef.current = null;
          }
        };
      }

      utt.onerror = (e) => {
        if (e.error !== 'canceled' && e.error !== 'interrupted') {
          if (activeIndexRef.current === index) {
            setSpeakingIndex(null);
            setIsPaused(false);
            activeIndexRef.current = null;
          }
        }
      };

      window.speechSynthesis.speak(utt);
    });
  }

  async function send(text = input) {
    if (!text.trim() || busy) return;
    stopSpeech();
    const next = [...messages, { role: 'user' as const, text }];
    setMessages(next);
    setInput('');
    setBusy(true);
    setError('');
    try {
      const r = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: messages.map(m => ({ role: m.role, content: m.text })),
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'The service could not answer.');
      setMessages([...next, { role: 'assistant', text: data.answer, citations: data.citations || [] }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Could not connect to the API.';
      if (msg.toLowerCase().includes('failed to fetch')) {
        setError('Unable to reach server. If using Render free tier, the server may be waking up from sleep (takes ~30-50s) — please wait a moment and try again.');
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  async function makeDraft() {
    setError('');
    try {
      const r = await fetch(`${API}/api/complaint/draft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ details }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Draft generation failed.');
      setDraft(d.draft);
      setShowDraft(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Draft generation failed.';
      if (msg.toLowerCase().includes('failed to fetch')) {
        setError('Unable to reach server. Please wait a moment while the server wakes up and try again.');
      } else {
        setError(msg);
      }
    }
  }

  function newChat() {
    stopSpeech();
    setMessages([]);
    setInput('');
    setError('');
  }

  const supportsTts = typeof window !== 'undefined' && 'speechSynthesis' in window;

  return (
    <main className="shell">
      <header>
        <div className="mark">CR</div>
        <div>
          <div className="brand">Consumer Rights AI</div>
          <div className="tag">Understand your consumer rights and explore possible remedies.</div>
        </div>
        <button className="quiet" onClick={newChat}>New conversation</button>
      </header>

      <section className="disclaimer">
        AI-generated general information. Verify sources and applicable law. This is not a substitute for advice from a qualified lawyer.
      </section>

      <section className="chat">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="eyebrow">CONSUMER SUPPORT</div>
            <h1>What happened?</h1>
            <p>Describe the purchase, service, or response you’re concerned about. Start with whatever details you have.</p>
            <div className="examples">
              {examples.map(x => (
                <button key={x} onClick={() => send(x)}>
                  {x}<span>↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <article key={i} className={`message ${m.role}`}>
              <div className="message-header">
                <div className="role">{m.role === 'user' ? 'You' : 'Consumer Rights AI'}</div>
                {m.role === 'assistant' && supportsTts && (
                  <div className="tts-controls">
                    {speakingIndex === i ? (
                      <>
                        <button
                          type="button"
                          className={`tts-btn ${!isPaused ? 'playing' : 'paused'}`}
                          onClick={() => speak(i, m.text)}
                          title={isPaused ? 'Resume reading aloud' : 'Pause reading aloud'}
                          aria-label={isPaused ? 'Resume reading aloud' : 'Pause reading aloud'}
                        >
                          {isPaused ? '▶ Resume' : '⏸ Pause'}
                        </button>
                        <button
                          type="button"
                          className="tts-btn stop"
                          onClick={stopSpeech}
                          title="Stop reading aloud"
                          aria-label="Stop reading aloud"
                        >
                          ⏹ Stop
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        className="tts-btn"
                        onClick={() => speak(i, m.text)}
                        title="Listen to response"
                        aria-label="Listen to response"
                      >
                        🔊 Listen
                      </button>
                    )}
                  </div>
                )}
              </div>
              <div className="body">{m.text}</div>
              {m.citations?.length ? (
                <details className="refs">
                  <summary>Sources ({m.citations.length})</summary>
                  {m.citations.map((c, j) => (
                    <div className="source" key={j}>
                      <b>{c.title}</b>
                      <small>
                        Dataset label: {c.verification_status} · {c.independently_verified ? 'Independent verification recorded' : 'Independent verification pending'} · {c.legal_provisions.join(', ')}
                      </small>
                      <p>{c.excerpt.slice(0, 320)}…</p>
                      {c.source_url && (
                        <a href={c.source_url} target="_blank" rel="noreferrer">Open source ↗</a>
                      )}
                    </div>
                  ))}
                </details>
              ) : null}
            </article>
          ))
        )}
        {busy && <div className="loading"><i /> Reviewing available sources…</div>}
        {error && <div className="error">{error}</div>}
      </section>

      <section className="resource-strip">
        <details>
          <summary>Evidence to gather</summary>
          <span>Invoice or receipt · order details · warranty or contract · screenshots · messages with the company · proof of payment</span>
        </details>
        <div className="official-links">
          <span>Official resources:</span>
          <a href="https://consumerhelpline.gov.in/public/" target="_blank" rel="noreferrer">National Consumer Helpline</a>
          <a href="https://e-jagriti.gov.in/" target="_blank" rel="noreferrer">e-Jagriti</a>
          <a href="https://irdai.gov.in/about-consumer-affairs" target="_blank" rel="noreferrer">IRDAI grievance help</a>
          <a href="https://www.dgca.gov.in/digigov-portal/" target="_blank" rel="noreferrer">DGCA</a>
        </div>
      </section>

      <section className="compose">
        <div className="composer">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Describe your consumer issue…"
            maxLength={12000}
          />
          <button onClick={() => send()} disabled={!input.trim() || busy}>
            Send <span>↑</span>
          </button>
        </div>
        <div className="actions">
          <span>Share only details you’re comfortable including.</span>
          <button onClick={() => { setDraft(''); setShowDraft(true); }}>
            Draft a complaint letter
          </button>
        </div>
      </section>

      {showDraft && (
        <div className="modal">
          <div className="dialog">
            <button className="close" onClick={() => setShowDraft(false)}>×</button>
            <h2>Complaint letter draft</h2>
            {draft ? (
              <>
                <p>Review and fill in all prompts before using.</p>
                <textarea value={draft} onChange={e => setDraft(e.target.value)} />
                <div className="modal-actions">
                  <button onClick={() => navigator.clipboard.writeText(draft)}>Copy draft</button>
                  <button
                    onClick={() => {
                      const a = document.createElement('a');
                      a.href = URL.createObjectURL(new Blob([draft], { type: 'text/plain' }));
                      a.download = 'consumer-complaint-draft.txt';
                      a.click();
                      URL.revokeObjectURL(a.href);
                    }}
                  >
                    Download
                  </button>
                </div>
              </>
            ) : (
              <>
                <p>Only information you provide will be included. Review the generated draft before using it.</p>
                <div className="draft-form">
                  {fields.map(([key, label, placeholder]) => (
                    <label key={key}>
                      {label}
                      {key === 'chronology' || key === 'evidence' ? (
                        <textarea
                          value={details[key] || ''}
                          placeholder={placeholder}
                          onChange={e => setDetails({ ...details, [key]: e.target.value })}
                        />
                      ) : (
                        <input
                          value={details[key] || ''}
                          placeholder={placeholder}
                          onChange={e => setDetails({ ...details, [key]: e.target.value })}
                        />
                      )}
                    </label>
                  ))}
                </div>
                <div className="modal-actions">
                  <button onClick={makeDraft}>Generate draft</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

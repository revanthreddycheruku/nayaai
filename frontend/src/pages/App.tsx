
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

declare global {
  interface Window {
    AndroidTTS?: {
      speak: (text: string) => void;
      stop: () => void;
    };
  }
}

function getApiUrl(): string {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/+$/, '');
  }

  if (typeof window !== 'undefined') {
    if (window.location.port === '5173') {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }

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
  [
    'remedy_requested',
    'What resolution do you want?',
    'For example, repair, refund, replacement',
  ],
  [
    'evidence',
    'Documents you can attach',
    'Invoice, screenshots, warranty, messages',
  ],
];

function cleanTextForSpeech(text: string): string {
  if (!text) return '';

  return text
    .replace(/₹\s*([0-9,]+)/g, '$1 rupees')
    .replace(/₹/g, ' rupees ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\[S\d+(?:,\s*S\d+)*\]/gi, '')
    .replace(/https?:\/\/\S+/gi, '')
    .replace(/^[ \t]*#+[ \t]*/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/^[ \t]*[-*_]{3,}[ \t]*$/gm, '')
    .replace(/^[ \t]*[-*+]\s+/gm, '')
    .replace(/^[ \t]*>[ \t]*/gm, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\n{2,}/g, '. ')
    .replace(/\n/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitTextIntoChunks(text: string, maxLen = 180): string[] {
  const sentences = text.match(/[^.!?\n]+[.!?\n]+/g) || [text];
  const chunks: string[] = [];
  let current = '';

  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;

    if ((current + ' ' + trimmed).trim().length <= maxLen) {
      current = current ? `${current} ${trimmed}` : trimmed;
    } else {
      if (current) chunks.push(current);

      if (trimmed.length > maxLen) {
        const words = trimmed.split(' ');
        let sub = '';

        for (const word of words) {
          if ((sub + ' ' + word).trim().length <= maxLen) {
            sub = sub ? `${sub} ${word}` : word;
          } else {
            if (sub) chunks.push(sub);
            sub = word;
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
  if (
    typeof window === 'undefined' ||
    !('speechSynthesis' in window)
  ) {
    return null;
  }

  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null;

  return (
    voices.find((voice) => voice.lang.toLowerCase() === 'en-in') ||
    voices.find(
      (voice) =>
        voice.lang.toLowerCase().startsWith('en') &&
        /natural|neural|online/i.test(voice.name),
    ) ||
    voices.find((voice) =>
      /^en-(us|gb)/i.test(voice.lang),
    ) ||
    voices.find((voice) =>
      voice.lang.toLowerCase().startsWith('en'),
    ) ||
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

  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const activeIndexRef = useRef<number | null>(null);

  const supportsTts =
    typeof window !== 'undefined' &&
    ('speechSynthesis' in window || !!window.AndroidTTS);

  useEffect(() => {
    if (
      typeof window !== 'undefined' &&
      'speechSynthesis' in window
    ) {
      window.speechSynthesis.getVoices();

      const onVoicesChanged = () => {
        window.speechSynthesis.getVoices();
      };

      window.speechSynthesis.addEventListener(
        'voiceschanged',
        onVoicesChanged,
      );

      return () => {
        window.speechSynthesis.cancel();
        window.speechSynthesis.removeEventListener(
          'voiceschanged',
          onVoicesChanged,
        );
      };
    }
  }, []);

  function stopSpeech() {
    if (typeof window !== 'undefined') {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }

      if (window.AndroidTTS) {
        try {
          window.AndroidTTS.stop();
        } catch (err) {
          console.error('Unable to stop Android speech:', err);
        }
      }
    }

    activeIndexRef.current = null;
    setSpeakingIndex(null);
    setIsPaused(false);
  }

  function speak(index: number, rawText: string) {
    if (typeof window === 'undefined') return;

    const cleaned = cleanTextForSpeech(rawText);
    if (!cleaned) return;

    // Use native Android text-to-speech inside the Android WebView.
    if (window.AndroidTTS) {
      if (speakingIndex === index) {
        stopSpeech();
        return;
      }

      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }

      activeIndexRef.current = index;
      setSpeakingIndex(index);
      setIsPaused(false);

      try {
        window.AndroidTTS.speak(cleaned);
      } catch (err) {
        console.error('Android text-to-speech failed:', err);
        activeIndexRef.current = null;
        setSpeakingIndex(null);
      }

      return;
    }

    // Otherwise, use the browser's speech synthesis.
    if (!('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported on this device.');
      return;
    }

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

    window.speechSynthesis.cancel();

    const chunks = splitTextIntoChunks(cleaned);
    const voice = getPreferredVoice();

    activeIndexRef.current = index;
    setSpeakingIndex(index);
    setIsPaused(false);

    chunks.forEach((chunk, chunkIndex) => {
      const utterance = new SpeechSynthesisUtterance(chunk);
      utterance.rate = 1;
      utterance.pitch = 1;

      if (voice) utterance.voice = voice;

      if (chunkIndex === chunks.length - 1) {
        utterance.onend = () => {
          if (activeIndexRef.current === index) {
            activeIndexRef.current = null;
            setSpeakingIndex(null);
            setIsPaused(false);
          }
        };
      }

      utterance.onerror = (event) => {
        if (
          event.error !== 'canceled' &&
          event.error !== 'interrupted' &&
          activeIndexRef.current === index
        ) {
          activeIndexRef.current = null;
          setSpeakingIndex(null);
          setIsPaused(false);
        }
      };

      window.speechSynthesis.speak(utterance);
    });
  }

  async function send(text = input) {
    if (!text.trim() || busy) return;

    stopSpeech();

    const next: Msg[] = [
      ...messages,
      { role: 'user', text },
    ];

    setMessages(next);
    setInput('');
    setBusy(true);
    setError('');

    try {
      const response = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: messages.map((message) => ({
            role: message.role,
            content: message.text,
          })),
        }),
      });

      if (!response.ok) {
        let errDetail = '';

        try {
          const errData = await response.json();
          errDetail = errData.detail || errData.message || '';
        } catch {
          errDetail = await response.text();
        }

        throw new Error(
          errDetail || `Server error (HTTP ${response.status})`,
        );
      }

      const data = await response.json();

      setMessages([
        ...next,
        {
          role: 'assistant',
          text: data.answer,
          citations: data.citations || [],
        },
      ]);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Could not connect to the API.';

      if (
        message.includes('Unexpected end of JSON input') ||
        message.toLowerCase().includes('failed to fetch')
      ) {
        setError(
          'Server is currently processing or waking up (about 30–50 seconds). Please wait a moment and try again.',
        );
      } else {
        setError(message);
      }
    } finally {
      setBusy(false);
    }
  }

  async function makeDraft() {
    setError('');

    try {
      const response = await fetch(`${API}/api/complaint/draft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ details }),
      });

      if (!response.ok) {
        let errDetail = '';

        try {
          const errData = await response.json();
          errDetail = errData.detail || errData.message || '';
        } catch {
          errDetail = await response.text();
        }

        throw new Error(
          errDetail ||
            `Draft generation failed (HTTP ${response.status})`,
        );
      }

      const data = await response.json();
      setDraft(data.draft);
      setShowDraft(true);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Draft generation failed.';

      if (
        message.includes('Unexpected end of JSON input') ||
        message.toLowerCase().includes('failed to fetch')
      ) {
        setError(
          'Unable to reach the server. Please wait a moment and try again.',
        );
      } else {
        setError(message);
      }
    }
  }

  function newChat() {
    stopSpeech();
    setMessages([]);
    setInput('');
    setError('');
  }

  return (
    <main className="shell">
      <header>
        <div className="mark">CR</div>

        <div>
          <div className="brand">Consumer Rights AI</div>
          <div className="tag">
            Understand your consumer rights and explore possible remedies.
          </div>
        </div>

        <button className="quiet" onClick={newChat}>
          New conversation
        </button>
      </header>

      <section className="disclaimer">
        AI-generated general information. Verify sources and applicable law.
        This is not a substitute for advice from a qualified lawyer.
      </section>

      <section className="chat">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="eyebrow">CONSUMER SUPPORT</div>
            <h1>What happened?</h1>
            <p>
              Describe the purchase, service, or response you’re concerned
              about. Start with whatever details you have.
            </p>

            <div className="examples">
              {examples.map((example) => (
                <button
                  key={example}
                  onClick={() => send(example)}
                >
                  {example}
                  <span>↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <article
              key={index}
              className={`message ${message.role}`}
            >
              <div className="message-header">
                <div className="role">
                  {message.role === 'user'
                    ? 'You'
                    : 'Consumer Rights AI'}
                </div>

                {message.role === 'assistant' && supportsTts && (
                  <div className="tts-controls">
                    {speakingIndex === index ? (
                      <button
                        type="button"
                        className="tts-btn stop"
                        onClick={stopSpeech}
                        title="Stop reading aloud"
                        aria-label="Stop reading aloud"
                      >
                        ⏹ Stop
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="tts-btn"
                        onClick={() => speak(index, message.text)}
                        title="Listen to response"
                        aria-label="Listen to response"
                      >
                        🔊 Listen
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div className="body">{message.text}</div>

              {message.citations?.length ? (
                <details className="refs">
                  <summary>
                    Sources ({message.citations.length})
                  </summary>

                  {message.citations.map((citation, citationIndex) => (
                    <div className="source" key={citationIndex}>
                      <b>{citation.title}</b>

                      <small>
                        Dataset label: {citation.verification_status}
                        {' · '}
                        {citation.independently_verified
                          ? 'Independent verification recorded'
                          : 'Independent verification pending'}
                        {' · '}
                        {citation.legal_provisions.join(', ')}
                      </small>

                      <p>{citation.excerpt.slice(0, 320)}…</p>

                      {citation.source_url && (
                        <a
                          href={citation.source_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open source ↗
                        </a>
                      )}
                    </div>
                  ))}
                </details>
              ) : null}
            </article>
          ))
        )}

        {busy && (
          <div className="loading">
            <i /> Reviewing available sources…
          </div>
        )}

        {error && <div className="error">{error}</div>}
      </section>

      <section className="resource-strip">
        <details>
          <summary>Evidence to gather</summary>
          <span>
            Invoice or receipt · order details · warranty or contract ·
            screenshots · messages with the company · proof of payment
          </span>
        </details>

        <div className="official-links">
          <span>Official resources:</span>

          <a
            href="https://consumerhelpline.gov.in/public/"
            target="_blank"
            rel="noreferrer"
          >
            National Consumer Helpline
          </a>

          <a
            href="https://e-jagriti.gov.in/"
            target="_blank"
            rel="noreferrer"
          >
            e-Jagriti
          </a>

          <a
            href="https://irdai.gov.in/about-consumer-affairs"
            target="_blank"
            rel="noreferrer"
          >
            IRDAI grievance help
          </a>

          <a
            href="https://www.dgca.gov.in/digigov-portal/"
            target="_blank"
            rel="noreferrer"
          >
            DGCA
          </a>
        </div>
      </section>

      <section className="compose">
        <div className="composer">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
            placeholder="Describe your consumer issue…"
            maxLength={12000}
          />

          <button
            onClick={() => send()}
            disabled={!input.trim() || busy}
          >
            Send <span>↑</span>
          </button>
        </div>

        <div className="actions">
          <span>
            Share only details you’re comfortable including.
          </span>

          <button
            onClick={() => {
              setDraft('');
              setShowDraft(true);
            }}
          >
            Draft a complaint letter
          </button>
        </div>
      </section>

      {showDraft && (
        <div className="modal">
          <div className="dialog">
            <button
              className="close"
              onClick={() => setShowDraft(false)}
            >
              ×
            </button>

            <h2>Complaint letter draft</h2>

            {draft ? (
              <>
                <p>Review and fill in all prompts before using.</p>

                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                />

                <div className="modal-actions">
                  <button
                    onClick={() =>
                      navigator.clipboard.writeText(draft)
                    }
                  >
                    Copy draft
                  </button>

                  <button
                    onClick={() => {
                      const url = URL.createObjectURL(
                        new Blob([draft], { type: 'text/plain' }),
                      );

                      const link = document.createElement('a');
                      link.href = url;
                      link.download = 'consumer-complaint-draft.txt';
                      link.click();

                      URL.revokeObjectURL(url);
                    }}
                  >
                    Download
                  </button>
                </div>
              </>
            ) : (
              <>
                <p>
                  Only information you provide will be included. Review
                  the generated draft before using it.
                </p>

                <div className="draft-form">
                  {fields.map(([key, label, placeholder]) => (
                    <label key={key}>
                      {label}

                      {key === 'chronology' || key === 'evidence' ? (
                        <textarea
                          value={details[key] || ''}
                          placeholder={placeholder}
                          onChange={(event) =>
                            setDetails({
                              ...details,
                              [key]: event.target.value,
                            })
                          }
                        />
                      ) : (
                        <input
                          value={details[key] || ''}
                          placeholder={placeholder}
                          onChange={(event) =>
                            setDetails({
                              ...details,
                              [key]: event.target.value,
                            })
                          }
                        />
                      )}
                    </label>
                  ))}
                </div>

                <div className="modal-actions">
                  <button onClick={makeDraft}>
                    Generate draft
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

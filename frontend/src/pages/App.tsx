
import { useCallback, useEffect, useRef, useState } from 'react';

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */

type Role = 'user' | 'assistant';

type Citation = {
  title?: string;
  source?: string;
  url?: string;
  link?: string;
  source_url?: string;
  case_name?: string;
  case_number?: string;
  court?: string;
  year?: number | string;
  snippet?: string;
  text?: string;
  excerpt?: string;
  legal_provisions?: string[];
  verification_status?: string;
  statute?: string;
  section?: string;
};

type Msg = {
  role: Role;
  text: string;
  citations?: Citation[];
};

type AndroidTTSBridge = {
  speak: (text: string) => void;
  stop: () => void;
};

type ChatResponse = {
  answer?: string;
  response?: string;
  message?: string;
  citations?: Citation[];
  sources?: Citation[];
  detail?: string;
};

type ComplaintResponse = {
  draft?: string;
  complaint?: string;
  text?: string;
  message?: string;
  detail?: string;
};

/* -------------------------------------------------------------------------- */
/* Android native text-to-speech bridge                                       */
/* -------------------------------------------------------------------------- */

function getAndroidTTS(): AndroidTTSBridge | undefined {
  if (typeof window === 'undefined') return undefined;

  return (
    window as unknown as {
      AndroidTTS?: AndroidTTSBridge;
    }
  ).AndroidTTS;
}

/* -------------------------------------------------------------------------- */
/* Configuration                                                              */
/* -------------------------------------------------------------------------- */

const API = (() => {
  const configuredUrl = import.meta.env.VITE_API_URL?.trim();

  if (configuredUrl) {
    return configuredUrl.replace(/\/+$/, '');
  }

  // During local Vite development, the backend runs on port 8000.
  if (
    typeof window !== 'undefined' &&
    window.location.port === '5173'
  ) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  // On a combined deployment, use the same origin.
  return '';
})();

const INITIAL_MESSAGE: Msg = {
  role: 'assistant',
  text:
    'Hello! I’m Naya AI, your consumer-rights information assistant. ' +
    'Describe your issue with a product, service, online purchase, refund, ' +
    'warranty, defective goods, or unfair trade practice. I can help you ' +
    'understand possible next steps and prepare a complaint draft.',
};

const OFFICIAL_RESOURCES = [
  {
    title: 'National Consumer Helpline',
    description: 'Consumer grievance information and assistance.',
    url: 'https://consumerhelpline.gov.in/',
  },
  {
    title: 'e-Jagriti',
    description: 'Consumer commission case filing and related services.',
    url: 'https://e-jagriti.gov.in/',
  },
  {
    title: 'Department of Consumer Affairs',
    description: 'Official consumer affairs information and updates.',
    url: 'https://consumeraffairs.nic.in/',
  },
];

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred. Please try again.';
}

function getAnswer(data: ChatResponse): string {
  const answer = data.answer ?? data.response ?? data.message;

  if (typeof answer === 'string' && answer.trim()) {
    return answer;
  }

  return 'I could not find a response. Please try rephrasing your question.';
}

function getCitations(data: ChatResponse): Citation[] {
  const items = data.citations ?? data.sources ?? [];
  return Array.isArray(items) ? items : [];
}

function cleanTextForSpeech(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' Code block omitted. ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '$1')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+[.)]\s+/gm, '')
    .replace(/[*_~>#]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitTextIntoChunks(text: string, maxLength = 220): string[] {
  const cleaned = cleanTextForSpeech(text);

  if (!cleaned) return [];

  const sentences = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [
    cleaned,
  ];

  const chunks: string[] = [];
  let current = '';

  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;

    if (trimmed.length > maxLength) {
      if (current) {
        chunks.push(current);
        current = '';
      }

      for (let i = 0; i < trimmed.length; i += maxLength) {
        chunks.push(trimmed.slice(i, i + maxLength));
      }

      continue;
    }

    if ((current + ' ' + trimmed).trim().length > maxLength) {
      if (current) chunks.push(current);
      current = trimmed;
    } else {
      current = `${current} ${trimmed}`.trim();
    }
  }

  if (current) chunks.push(current);

  return chunks;
}

function getPreferredVoice(): SpeechSynthesisVoice | undefined {
  if (
    typeof window === 'undefined' ||
    !('speechSynthesis' in window)
  ) {
    return undefined;
  }

  const voices = window.speechSynthesis.getVoices();

  return (
    voices.find((voice) => voice.lang.toLowerCase() === 'en-in') ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith('en-')) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith('en'))
  );
}

function getCitationTitle(citation: Citation): string {
  return (
    citation.title ??
    citation.case_name ??
    citation.source ??
    'Source'
  );
}

function getCitationUrl(citation: Citation): string | undefined {
  const candidate = citation.source_url ?? citation.url ?? citation.link;

  if (!candidate) return undefined;

  try {
    const parsed = new URL(candidate);
    if (parsed.protocol === 'https:' || parsed.protocol === 'http:') {
      return parsed.href;
    }
  } catch {
    // The supplied citation URL is not a valid absolute URL.
  }

  return undefined;
}

/* -------------------------------------------------------------------------- */
/* Main app                                                                   */
/* -------------------------------------------------------------------------- */

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  const [complaintDetails, setComplaintDetails] = useState({
    consumer_name: '',
    company_name: '',
    product_or_service: '',
    issue: '',
    purchase_date: '',
    amount: '',
    desired_resolution: '',
  });

  const [complaintDraft, setComplaintDraft] = useState('');
  const [complaintBusy, setComplaintBusy] = useState(false);
  const [complaintError, setComplaintError] = useState('');

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const activeIndexRef = useRef<number | null>(null);
  const speechRunRef = useRef(0);

  const supportsTts =
    typeof window !== 'undefined' &&
    (Boolean(getAndroidTTS()) ||
      ('speechSynthesis' in window &&
        'SpeechSynthesisUtterance' in window));

  /* ------------------------------ Scrolling ------------------------------ */

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  /* ------------------------------ Speech --------------------------------- */

  const stopSpeech = useCallback(() => {
    speechRunRef.current += 1;

    if (
      typeof window !== 'undefined' &&
      'speechSynthesis' in window
    ) {
      window.speechSynthesis.cancel();
    }

    getAndroidTTS()?.stop();

    activeIndexRef.current = null;
    setSpeakingIndex(null);
    setIsPaused(false);
  }, []);

  useEffect(() => {
    return () => {
      if (
        typeof window !== 'undefined' &&
        'speechSynthesis' in window
      ) {
        window.speechSynthesis.cancel();
      }

      getAndroidTTS()?.stop();
    };
  }, []);

  const speak = useCallback(
    (index: number, rawText: string) => {
      const text = cleanTextForSpeech(rawText);
      if (!text) return;

      // Prefer Android's native TextToSpeech engine inside the WebView.
      const androidTTS = getAndroidTTS();

      if (androidTTS) {
        if (activeIndexRef.current === index) {
          stopSpeech();
          return;
        }

        stopSpeech();
        activeIndexRef.current = index;
        setSpeakingIndex(index);
        setIsPaused(false);

        try {
          androidTTS.speak(text);
        } catch {
          activeIndexRef.current = null;
          setSpeakingIndex(null);
          setError('Unable to start Android speech. Please try again.');
        }

        return;
      }

      // Browser speech fallback.
      if (
        typeof window === 'undefined' ||
        !('speechSynthesis' in window) ||
        !('SpeechSynthesisUtterance' in window)
      ) {
        setError('Text-to-speech is not available on this device.');
        return;
      }

      if (activeIndexRef.current === index) {
        stopSpeech();
        return;
      }

      stopSpeech();

      const runId = speechRunRef.current;
      const chunks = splitTextIntoChunks(text);

      if (chunks.length === 0) return;

      activeIndexRef.current = index;
      setSpeakingIndex(index);
      setIsPaused(false);

      const speakChunk = (chunkIndex: number) => {
        if (
          runId !== speechRunRef.current ||
          activeIndexRef.current !== index
        ) {
          return;
        }

        if (chunkIndex >= chunks.length) {
          activeIndexRef.current = null;
          setSpeakingIndex(null);
          setIsPaused(false);
          return;
        }

        const utterance = new SpeechSynthesisUtterance(
          chunks[chunkIndex],
        );

        utterance.lang = 'en-IN';
        utterance.rate = 1;
        utterance.pitch = 1;

        const voice = getPreferredVoice();
        if (voice) utterance.voice = voice;

        utterance.onend = () => {
          if (
            runId === speechRunRef.current &&
            activeIndexRef.current === index
          ) {
            speakChunk(chunkIndex + 1);
          }
        };

        utterance.onerror = (event) => {
          if (
            event.error !== 'canceled' &&
            event.error !== 'interrupted' &&
            runId === speechRunRef.current &&
            activeIndexRef.current === index
          ) {
            activeIndexRef.current = null;
            setSpeakingIndex(null);
            setIsPaused(false);
          }
        };

        window.speechSynthesis.speak(utterance);
      };

      speakChunk(0);
    },
    [stopSpeech],
  );

  const togglePause = () => {
    if (
      typeof window === 'undefined' ||
      !('speechSynthesis' in window) ||
      getAndroidTTS()
    ) {
      return;
    }

    if (window.speechSynthesis.speaking) {
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        setIsPaused(false);
      } else {
        window.speechSynthesis.pause();
        setIsPaused(true);
      }
    }
  };

  /* ------------------------------ Chat ------------------------------------ */

  async function send(text = input) {
    const messageText = text.trim();

    if (!messageText || busy) return;

    stopSpeech();
    setError('');

    const previousMessages = messages.filter(
      (message) => message !== INITIAL_MESSAGE,
    );

    setMessages((current) => [
      ...current,
      { role: 'user', text: messageText },
    ]);

    setInput('');
    setBusy(true);

    try {
      const response = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText,
          history: previousMessages.map((message) => ({
            role: message.role,
            content: message.text,
          })),
        }),
      });

      if (!response.ok) {
        let errDetail = '';

        try {
          const errData = await response.json();
          errDetail =
            errData.detail ??
            errData.message ??
            errData.error ??
            '';
        } catch {
          errDetail = await response.text();
        }

        throw new Error(
          errDetail || `Server error (HTTP ${response.status})`,
        );
      }

      const data = (await response.json()) as ChatResponse;

      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          text: getAnswer(data),
          citations: getCitations(data),
        },
      ]);
    } catch (err) {
      setError(
        `${getErrorMessage(err)} Check your internet connection and backend URL.`,
      );
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void send();
  }

  function handleInputKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      void send();
    }
  }

  function startNewChat() {
    stopSpeech();
    setMessages([INITIAL_MESSAGE]);
    setInput('');
    setError('');
    setComplaintDraft('');
    setComplaintError('');
  }

  /* ------------------------- Complaint generation ----------------------- */

  function updateComplaintField(
    field: keyof typeof complaintDetails,
    value: string,
  ) {
    setComplaintDetails((current) => ({
      ...current,
      [field]: value,
    }));
  }

  async function generateComplaint(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!complaintDetails.company_name.trim() ||
        !complaintDetails.issue.trim()) {
      setComplaintError(
        'Please enter the company name and describe your issue.',
      );
      return;
    }

    setComplaintBusy(true);
    setComplaintError('');
    setComplaintDraft('');

    try {
      const response = await fetch(`${API}/api/complaint/draft`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(complaintDetails),
      });

      if (!response.ok) {
        let detail = '';

        try {
          const data = await response.json();
          detail = data.detail ?? data.message ?? '';
        } catch {
          detail = await response.text();
        }

        throw new Error(
          detail || `Server error (HTTP ${response.status})`,
        );
      }

      const data = (await response.json()) as ComplaintResponse;

      const draft =
        data.draft ?? data.complaint ?? data.text ?? data.message;

      if (!draft) {
        throw new Error(
          'The server did not return a complaint draft. Check the backend response format.',
        );
      }

      setComplaintDraft(draft);
    } catch (err) {
      setComplaintError(getErrorMessage(err));
    } finally {
      setComplaintBusy(false);
    }
  }

  async function copyText(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';

      document.body.appendChild(textArea);
      textArea.select();

      const copied = document.execCommand('copy');
      document.body.removeChild(textArea);

      if (!copied) {
        setError('Unable to copy automatically. Please select and copy the text.');
      }
    }
  }

  /* ------------------------------ UI ------------------------------------- */

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-400 font-bold text-slate-950">
              N
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">
                Naya AI
              </h1>
              <p className="text-xs text-slate-400">
                Consumer Rights Assistant
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={startNewChat}
            className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
          >
            New chat
          </button>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="flex min-h-[70vh] min-w-0 flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/60">
          <div className="border-b border-slate-800 px-5 py-4">
            <h2 className="font-semibold">Ask about your consumer issue</h2>
            <p className="mt-1 text-sm text-slate-400">
              Explain what happened and ask about possible next steps.
            </p>
          </div>

          <div
            className="flex-1 space-y-5 overflow-y-auto p-4 sm:p-6"
            aria-live="polite"
          >
            {messages.map((message, index) => (
              <article
                key={`${index}-${message.role}`}
                className={`flex gap-3 ${
                  message.role === 'user' ? 'flex-row-reverse' : ''
                }`}
              >
                <div
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
                    message.role === 'user'
                      ? 'bg-indigo-400 text-slate-950'
                      : 'bg-teal-400 text-slate-950'
                  }`}
                >
                  {message.role === 'user' ? 'You' : 'N'}
                </div>

                <div
                  className={`min-w-0 max-w-[90%] rounded-2xl px-4 py-3 sm:max-w-[85%] ${
                    message.role === 'user'
                      ? 'bg-indigo-500/15 text-slate-100'
                      : 'bg-slate-800/80 text-slate-100'
                  }`}
                >
                  <div className="whitespace-pre-wrap break-words text-sm leading-7">
                    {message.text}
                  </div>

                  {message.role === 'assistant' && (
                    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-700 pt-3">
                      {supportsTts && (
                        <>
                          <button
                            type="button"
                            onClick={() => speak(index, message.text)}
                            className="rounded-lg border border-slate-600 px-3 py-1.5 text-xs transition hover:bg-slate-700"
                          >
                            {speakingIndex === index ? 'Stop' : 'Listen'}
                          </button>

                          {speakingIndex === index &&
                            !getAndroidTTS() && (
                              <button
                                type="button"
                                onClick={togglePause}
                                className="rounded-lg border border-slate-600 px-3 py-1.5 text-xs transition hover:bg-slate-700"
                              >
                                {isPaused ? 'Resume' : 'Pause'}
                              </button>
                            )}
                        </>
                      )}

                      <button
                        type="button"
                        onClick={() => void copyText(message.text)}
                        className="rounded-lg border border-slate-600 px-3 py-1.5 text-xs transition hover:bg-slate-700"
                      >
                        Copy answer
                      </button>
                    </div>
                  )}

                  {message.citations && message.citations.length > 0 && (
                    <div className="mt-4 border-t border-slate-700 pt-3">
                      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-teal-300">
                        Sources
                      </h3>

                      <ul className="space-y-2">
                        {message.citations.map((citation, citationIndex) => {
                          const url = getCitationUrl(citation);

                          return (
                            <li
                              key={`${citationIndex}-${getCitationTitle(citation)}`}
                              className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-xs"
                            >
                              {url ? (
                                <a
                                  href={url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="font-medium text-teal-300 underline underline-offset-2 hover:text-teal-200"
                                >
                                  {getCitationTitle(citation)}
                                </a>
                              ) : (
                                <p className="font-medium text-slate-200">
                                  {getCitationTitle(citation)}
                                </p>
                              )}

                              {(citation.court ||
                                citation.year ||
                                citation.case_number) && (
                                <p className="mt-1 text-slate-400">
                                  {[
                                    citation.court,
                                    citation.year,
                                    citation.case_number,
                                  ]
                                    .filter(Boolean)
                                    .join(' · ')}
                                </p>
                              )}

                              {(citation.excerpt || citation.snippet || citation.text) && (
                                <p className="mt-2 leading-5 text-slate-400">
                                  {citation.excerpt ?? citation.snippet ?? citation.text}
                                </p>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </div>
              </article>
            ))}

            {busy && (
              <div className="flex items-center gap-3 text-sm text-slate-400">
                <div className="h-2 w-2 animate-pulse rounded-full bg-teal-400" />
                Naya AI is preparing a response…
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {error && (
            <div
              role="alert"
              className="mx-4 mb-3 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200"
            >
              {error}
              <button
                type="button"
                onClick={() => setError('')}
                className="ml-3 underline"
              >
                Dismiss
              </button>
            </div>
          )}

          <form
            onSubmit={handleSubmit}
            className="border-t border-slate-800 p-4 sm:p-5"
          >
            <label htmlFor="chat-input" className="sr-only">
              Describe your consumer issue
            </label>

            <div className="flex items-end gap-2">
              <textarea
                ref={inputRef}
                id="chat-input"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="For example: The seller refused to refund my defective product…"
                rows={2}
                maxLength={8000}
                disabled={busy}
                className="max-h-40 min-h-12 flex-1 resize-y rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm outline-none transition placeholder:text-slate-500 focus:border-teal-400 disabled:opacity-60"
              />

              <button
                type="submit"
                disabled={busy || !input.trim()}
                className="rounded-xl bg-teal-400 px-4 py-3 text-sm font-bold text-slate-950 transition hover:bg-teal-300 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {busy ? 'Sending…' : 'Send'}
              </button>
            </div>

            <p className="mt-2 text-xs text-slate-500">
              Press Enter to send · Shift+Enter for a new line
            </p>
          </form>
        </section>

        <aside className="space-y-5">
          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
            <h2 className="text-base font-semibold">What can I help with?</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Ask questions about refunds, defective products, misleading
              advertisements, warranty disputes, online shopping, and
              consumer complaints.
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              {[
                'A seller refuses my refund',
                'I received a defective product',
                'A company denied my warranty',
                'How do I file a complaint?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  disabled={busy}
                  onClick={() => void send(suggestion)}
                  className="rounded-full border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 transition hover:border-teal-400 hover:text-teal-300 disabled:opacity-50"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </section>

          <section
            id="complaint-builder"
            className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"
          >
            <h2 className="text-base font-semibold">Complaint draft builder</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Enter the relevant details to request a written complaint draft.
            </p>

            <form
              onSubmit={generateComplaint}
              className="mt-4 space-y-3"
            >
              <div>
                <label
                  htmlFor="consumer-name"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Your name (optional)
                </label>
                <input
                  id="consumer-name"
                  value={complaintDetails.consumer_name}
                  onChange={(event) =>
                    updateComplaintField('consumer_name', event.target.value)
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                  placeholder="Consumer name"
                />
              </div>

              <div>
                <label
                  htmlFor="company-name"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Company or seller *
                </label>
                <input
                  id="company-name"
                  required
                  value={complaintDetails.company_name}
                  onChange={(event) =>
                    updateComplaintField('company_name', event.target.value)
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                  placeholder="Company or seller name"
                />
              </div>

              <div>
                <label
                  htmlFor="product-name"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Product or service
                </label>
                <input
                  id="product-name"
                  value={complaintDetails.product_or_service}
                  onChange={(event) =>
                    updateComplaintField(
                      'product_or_service',
                      event.target.value,
                    )
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                  placeholder="Product or service"
                />
              </div>

              <div>
                <label
                  htmlFor="issue-description"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Describe the issue *
                </label>
                <textarea
                  id="issue-description"
                  required
                  rows={4}
                  value={complaintDetails.issue}
                  onChange={(event) =>
                    updateComplaintField('issue', event.target.value)
                  }
                  className="w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                  placeholder="Explain what happened, including any attempts to resolve it."
                />
              </div>

              <div>
                <label
                  htmlFor="purchase-date"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Purchase date
                </label>
                <input
                  id="purchase-date"
                  type="date"
                  value={complaintDetails.purchase_date}
                  onChange={(event) =>
                    updateComplaintField('purchase_date', event.target.value)
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                />
              </div>

              <div>
                <label
                  htmlFor="purchase-amount"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Amount paid (₹)
                </label>
                <input
                  id="purchase-amount"
                  inputMode="decimal"
                  value={complaintDetails.amount}
                  onChange={(event) =>
                    updateComplaintField('amount', event.target.value)
                  }
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                  placeholder="Amount paid"
                />
              </div>

              <div>
                <label
                  htmlFor="desired-resolution"
                  className="mb-1 block text-xs text-slate-300"
                >
                  Desired resolution
                </label>
                <textarea
                  id="desired-resolution"
                  rows={2}
                  value={complaintDetails.desired_resolution}
                  onChange={(event) =>
                    updateComplaintField(
                      'desired_resolution',
                      event.target.value,
                    )
                  }
                  className="w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-teal-400"
                  placeholder="Refund, replacement, repair, or other resolution"
                />
              </div>

              {complaintError && (
                <p role="alert" className="text-sm text-red-300">
                  {complaintError}
                </p>
              )}

              <button
                type="submit"
                disabled={complaintBusy}
                className="w-full rounded-xl bg-teal-400 px-4 py-3 text-sm font-bold text-slate-950 transition hover:bg-teal-300 disabled:opacity-50"
              >
                {complaintBusy ? 'Generating draft…' : 'Generate complaint draft'}
              </button>
            </form>

            {complaintDraft && (
              <div className="mt-5 border-t border-slate-700 pt-4">
                <h3 className="font-semibold">Your draft</h3>

                <div className="mt-3 max-h-96 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm leading-6">
                  {complaintDraft}
                </div>

                <button
                  type="button"
                  onClick={() => void copyText(complaintDraft)}
                  className="mt-3 w-full rounded-lg border border-slate-600 px-3 py-2 text-sm transition hover:bg-slate-800"
                >
                  Copy complaint
                </button>

                <p className="mt-3 text-xs leading-5 text-amber-200/80">
                  Review all details and supporting evidence before submitting
                  the complaint. This draft is not a filed legal complaint.
                </p>
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
            <h2 className="text-base font-semibold">Official resources</h2>

            <div className="mt-3 space-y-3">
              {OFFICIAL_RESOURCES.map((resource) => (
                <a
                  key={resource.url}
                  href={resource.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block rounded-xl border border-slate-800 p-3 transition hover:border-teal-400"
                >
                  <span className="text-sm font-medium text-teal-300">
                    {resource.title} ↗
                  </span>
                  <span className="mt-1 block text-xs leading-5 text-slate-400">
                    {resource.description}
                  </span>
                </a>
              ))}
            </div>
          </section>

          <section className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4">
            <h2 className="text-sm font-semibold text-amber-200">
              Important disclaimer
            </h2>
            <p className="mt-2 text-xs leading-5 text-slate-400">
              Naya AI provides general legal information, not a substitute
              for advice from a qualified lawyer. AI-generated answers and
              citations may be incomplete or inaccurate. Verify applicable
              law, case details, deadlines, and official procedures before
              acting.
            </p>
          </section>
        </aside>
      </main>

      <footer className="border-t border-slate-800 px-4 py-5 text-center text-xs text-slate-500">
        Naya AI · Consumer Rights Information Assistant
      </footer>
    </div>
  );
}

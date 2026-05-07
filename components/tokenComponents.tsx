'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Anthropic from '@anthropic-ai/sdk';
import { Textarea } from "@/components/ui/textarea";
import { Upload, FileText, Image as ImageIcon, X, ChevronDown, Key } from "lucide-react";

const debounce = <T extends (...args: any[]) => void>(func: T, delay: number) => {
    let timeoutId: ReturnType<typeof setTimeout>;
    return (...args: Parameters<T>) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
    };
};

async function getGPT4oTokenCount(text: string): Promise<number | null> {
    try {
        const { encodingForModel } = await import('js-tiktoken');
        const encoder = encodingForModel('gpt-4o');
        return encoder.encode(text).length;
    } catch {
        return null;
    }
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.length; i += 8192) {
        binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + 8192, bytes.length)));
    }
    return btoa(binary);
}

const CLAUDE_MODELS = [
    { id: 'claude-opus-4-7', name: 'Claude Opus 4.7', inputPricePerMTok: 15 },
    { id: 'claude-opus-4-6', name: 'Claude Opus 4.6', inputPricePerMTok: 15 },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', inputPricePerMTok: 3 },
    { id: 'claude-sonnet-4-5-20250929', name: 'Claude Sonnet 4.5', inputPricePerMTok: 3 },
    { id: 'claude-opus-4-1-20250805', name: 'Claude Opus 4.1', inputPricePerMTok: 15 },
    { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', inputPricePerMTok: 1 },
    { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4', inputPricePerMTok: 3 },
    { id: 'claude-opus-4-20250514', name: 'Claude Opus 4', inputPricePerMTok: 15 },
    { id: 'claude-3-7-sonnet-20250219', name: 'Claude 3.7 Sonnet', inputPricePerMTok: 3 },
];

const GPT4O_INPUT_PRICE_PER_MTOK = 2.5;

const formatCost = (tokens: number, pricePerMTok: number): string => {
    const cost = (tokens / 1_000_000) * pricePerMTok;
    if (cost === 0) return '$0.00';
    if (cost < 0.01) return `$${cost.toFixed(5)}`;
    if (cost < 1) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
};

const OPUS_47_ID = 'claude-opus-4-7';
const OPUS_46_ID = 'claude-opus-4-6';
const OPUS_46_NAME = 'Claude Opus 4.6';

const ACCEPTED_FILE_TYPES = {
    image: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
    pdf: ['.pdf'],
    text: ['.txt', '.md', '.js', '.jsx', '.ts', '.tsx', '.json', '.html', '.css', '.csv'],
};

const getAcceptedFileTypes = () => [
    ...ACCEPTED_FILE_TYPES.image,
    ...ACCEPTED_FILE_TYPES.pdf,
    ...ACCEPTED_FILE_TYPES.text,
].join(',');

const getFileTypeCategory = (file: File): 'image' | 'pdf' | 'text' | 'unknown' => {
    if (file.type.startsWith('image/')) return 'image';
    if (file.type === 'application/pdf') return 'pdf';
    if (
        file.type.includes('text') ||
        file.type.includes('javascript') ||
        file.type.includes('json') ||
        file.type.includes('html') ||
        file.type.includes('css') ||
        file.name.endsWith('.md') ||
        file.name.endsWith('.csv')
    ) return 'text';
    return 'unknown';
};

const API_KEY_STORAGE_KEY = 'anthropic_api_key';

export const TokenizerInput = () => {
    const [apiKey, setApiKey] = useState('');
    const [apiKeyInput, setApiKeyInput] = useState('');
    const [showApiKeyInput, setShowApiKeyInput] = useState(false);
    const [text, setText] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [fileType, setFileType] = useState<'image' | 'pdf' | 'text' | 'unknown'>('unknown');
    const [filePreview, setFilePreview] = useState<string | null>(null);
    const [selectedModel, setSelectedModel] = useState(CLAUDE_MODELS[0].id);
    const [stats, setStats] = useState<{
        tokens: number | null;
        gpt4oTokens: number | null;
        comparisonTokens: number | null;
        comparisonModel: string | null;
        chars: number;
        fileName?: string;
    }>({ tokens: null, gpt4oTokens: null, comparisonTokens: null, comparisonModel: null, chars: 0 });
    const [error, setError] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [showModelDropdown, setShowModelDropdown] = useState(false);

    useEffect(() => {
        const stored = localStorage.getItem(API_KEY_STORAGE_KEY);
        if (stored) {
            setApiKey(stored);
            setApiKeyInput(stored);
        } else {
            setShowApiKeyInput(true);
        }
    }, []);

    const saveApiKey = () => {
        const trimmed = apiKeyInput.trim();
        if (!trimmed) return;
        localStorage.setItem(API_KEY_STORAGE_KEY, trimmed);
        setApiKey(trimmed);
        setShowApiKeyInput(false);
    };

    const clearApiKey = () => {
        localStorage.removeItem(API_KEY_STORAGE_KEY);
        setApiKey('');
        setApiKeyInput('');
        setShowApiKeyInput(true);
    };

    // analyzeRef holds the latest version of handleAnalyzeText so the stable
    // debounced wrapper always calls up-to-date logic without being recreated.
    const analyzeRef = useRef<(t: string) => void>(() => {});

    const handleAnalyzeText = useCallback(async (inputText: string) => {
        if (!inputText.trim()) {
            setStats({ tokens: null, gpt4oTokens: null, comparisonTokens: null, comparisonModel: null, chars: 0 });
            setError(null);
            return;
        }
        if (!apiKey) {
            setError('Please enter your Anthropic API key first.');
            return;
        }

        setIsProcessing(true);
        setError(null);

        try {
            const comparisonModel = selectedModel === OPUS_47_ID ? OPUS_46_ID : null;
            const betas: string[] = ['token-counting-2024-11-01'];
            const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });

            const [mainCount, compCount, gpt4oTokens] = await Promise.all([
                client.beta.messages.countTokens({ betas, model: selectedModel, messages: [{ role: 'user', content: inputText }] }),
                comparisonModel
                    ? client.beta.messages.countTokens({ betas, model: comparisonModel, messages: [{ role: 'user', content: inputText }] })
                    : Promise.resolve(null),
                getGPT4oTokenCount(inputText),
            ]);

            const adjust = (t: number) => (t > 7 ? t - 7 : 0);

            setStats({
                tokens: adjust(mainCount.input_tokens),
                gpt4oTokens,
                comparisonTokens: compCount != null
                    ? (compCount.input_tokens > 7 ? compCount.input_tokens - 7 : compCount.input_tokens)
                    : null,
                comparisonModel,
                chars: inputText.length,
            });
        } catch (err: any) {
            console.error('Token counting error:', err);
            const msg = err?.error?.message || err?.message || 'Failed to analyze text. Please try again.';
            setError(msg);
            setStats({ tokens: null, gpt4oTokens: null, comparisonTokens: null, comparisonModel: null, chars: inputText.length });
        } finally {
            setIsProcessing(false);
        }
    }, [selectedModel, apiKey]);

    useEffect(() => { analyzeRef.current = handleAnalyzeText; }, [handleAnalyzeText]);

    // Stable debounced wrapper — created once, always delegates to analyzeRef.
    const debouncedAnalyze = useRef(
        debounce((t: string) => analyzeRef.current(t), 500)
    ).current;

    useEffect(() => {
        if (!file && text) debouncedAnalyze(text);
    }, [text, file, debouncedAnalyze]);

    const handleAnalyzeFile = async () => {
        if (!file || !apiKey) return;
        setIsProcessing(true);
        setError(null);
        try {
            const arrayBuffer = await file.arrayBuffer();
            const base64Content = arrayBufferToBase64(arrayBuffer);
            const comparisonModel = selectedModel === OPUS_47_ID ? OPUS_46_ID : null;
            const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });

            let messages: any[];
            let betas: string[];

            if (fileType === 'pdf') {
                betas = ['token-counting-2024-11-01', 'pdfs-2024-09-25'];
                messages = [{
                    role: 'user',
                    content: [{ type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: base64Content } }],
                }];
            } else if (fileType === 'image') {
                betas = ['token-counting-2024-11-01'];
                const mediaType = (file.type || 'image/jpeg') as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp';
                messages = [{
                    role: 'user',
                    content: [{ type: 'image', source: { type: 'base64', media_type: mediaType, data: base64Content } }],
                }];
            } else {
                betas = ['token-counting-2024-11-01'];
                const text = new TextDecoder().decode(new Uint8Array(arrayBuffer));
                messages = [{ role: 'user', content: text }];
            }

            const [count, comparison] = await Promise.all([
                client.beta.messages.countTokens({ betas, model: selectedModel, messages }),
                comparisonModel
                    ? client.beta.messages.countTokens({ betas, model: comparisonModel, messages })
                    : Promise.resolve(null),
            ]);

            setStats({
                tokens: count.input_tokens > 7 ? count.input_tokens - 7 : 0,
                gpt4oTokens: null,
                comparisonTokens: comparison
                    ? (comparison.input_tokens > 7 ? comparison.input_tokens - 7 : comparison.input_tokens)
                    : null,
                comparisonModel,
                chars: new Uint8Array(arrayBuffer).length,
                fileName: file.name,
            });
        } catch (err: any) {
            console.error('Token counting error:', err);
            setError(err?.error?.message || err?.message || 'Failed to analyze file. Please try again.');
            setStats({ tokens: null, gpt4oTokens: null, comparisonTokens: null, comparisonModel: null, chars: 0 });
        } finally {
            setIsProcessing(false);
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = event.target.files?.[0] || null;
        if (!selectedFile) return;
        const type = getFileTypeCategory(selectedFile);
        setFile(selectedFile);
        setFileType(type);
        setText('');
        setStats({ tokens: null, gpt4oTokens: null, comparisonTokens: null, comparisonModel: null, chars: 0, fileName: selectedFile.name });
        if (type === 'image') {
            const reader = new FileReader();
            reader.onload = (e) => setFilePreview(e.target?.result as string);
            reader.readAsDataURL(selectedFile);
        } else {
            setFilePreview(null);
        }
    };

    const clearFile = () => {
        setFile(null);
        setFileType('unknown');
        setFilePreview(null);
        setStats({ tokens: null, gpt4oTokens: null, comparisonTokens: null, comparisonModel: null, chars: 0 });
        const fileInput = document.getElementById('file-upload') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
    };

    const selectModel = (modelId: string) => {
        setSelectedModel(modelId);
        setShowModelDropdown(false);
    };

    const selectedModelInfo = CLAUDE_MODELS.find(m => m.id === selectedModel);
    const selectedModelName = selectedModelInfo?.name || selectedModel;
    const selectedModelPrice = selectedModelInfo?.inputPricePerMTok ?? null;
    const comparisonModelInfo = stats.comparisonModel ? CLAUDE_MODELS.find(m => m.id === stats.comparisonModel) : null;
    const comparisonModelPrice = comparisonModelInfo?.inputPricePerMTok ?? null;

    return (
        <div className="flex flex-col space-y-4 max-w-3xl mx-auto">
            {/* API Key */}
            <div className="rounded-xl border border-neutral-700 bg-neutral-800 p-4">
                {!showApiKeyInput && apiKey ? (
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-sm text-neutral-400">
                            <Key size={14} />
                            <span>API key: <span className="font-mono text-neutral-300">{apiKey.slice(0, 8)}…{apiKey.slice(-4)}</span></span>
                        </div>
                        <button
                            onClick={clearApiKey}
                            className="text-xs text-neutral-500 hover:text-neutral-300 underline"
                        >
                            Change
                        </button>
                    </div>
                ) : (
                    <div className="space-y-3">
                        <div className="flex items-center gap-2 text-sm text-neutral-400">
                            <Key size={14} />
                            <span>Enter your Anthropic API key to count Claude tokens</span>
                        </div>
                        <div className="flex gap-2">
                            <input
                                type="password"
                                placeholder="sk-ant-..."
                                value={apiKeyInput}
                                onChange={(e) => setApiKeyInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && saveApiKey()}
                                className="flex-1 rounded-md bg-neutral-900 border border-neutral-600 px-3 py-2 text-sm font-mono focus:outline-none focus:border-neutral-400"
                            />
                            <button
                                onClick={saveApiKey}
                                disabled={!apiKeyInput.trim()}
                                className="rounded-md bg-neutral-700 px-4 py-2 text-sm hover:bg-neutral-600 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                Save
                            </button>
                        </div>
                        <p className="text-xs text-neutral-600">
                            Your key is stored only in your browser's localStorage and never sent to any third-party server.
                        </p>
                    </div>
                )}
            </div>

            {/* Model selector */}
            <div className="flex justify-end mb-2 relative">
                <div
                    className="flex items-center gap-2 cursor-pointer rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm hover:bg-neutral-700"
                    onClick={() => setShowModelDropdown(!showModelDropdown)}
                >
                    <span>{selectedModelName}</span>
                    <ChevronDown size={16} className={`transition-transform ${showModelDropdown ? 'rotate-180' : ''}`} />
                </div>
                {showModelDropdown && (
                    <div className="absolute right-0 top-full mt-1 w-64 rounded-md border border-neutral-700 bg-neutral-800 shadow-lg z-10">
                        <div className="py-1">
                            {CLAUDE_MODELS.map((model) => (
                                <div
                                    key={model.id}
                                    className={`px-4 py-2 text-sm cursor-pointer hover:bg-neutral-700 ${model.id === selectedModel ? 'bg-neutral-700' : ''}`}
                                    onClick={() => selectModel(model.id)}
                                >
                                    {model.name}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* File upload */}
            <div className="flex items-center justify-between gap-4 mb-3">
                <div className="flex items-center gap-2">
                    <div className="relative">
                        <input
                            type="file"
                            id="file-upload"
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            onChange={handleFileChange}
                            accept={getAcceptedFileTypes()}
                        />
                        <button className="flex items-center gap-2 rounded-md bg-neutral-800 px-3 py-2 text-sm hover:bg-neutral-700 border border-neutral-700">
                            <Upload size={16} />
                            Upload File
                        </button>
                    </div>
                    {file && (
                        <div className="flex items-center rounded-md bg-neutral-800 border border-neutral-700 px-3 py-1.5 text-sm">
                            {fileType === 'image' && <ImageIcon size={14} className="mr-2" />}
                            {(fileType === 'pdf' || fileType === 'text') && <FileText size={14} className="mr-2" />}
                            <span className="truncate max-w-[150px]">{file.name}</span>
                            <button onClick={clearFile} className="ml-2 text-neutral-400 hover:text-white">
                                <X size={14} />
                            </button>
                        </div>
                    )}
                </div>
                {file && (
                    <button
                        onClick={handleAnalyzeFile}
                        disabled={isProcessing || !apiKey}
                        className="whitespace-nowrap rounded-md bg-neutral-800 px-4 py-2 text-sm hover:bg-neutral-700 border border-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isProcessing ? 'Processing…' : 'Count Tokens'}
                    </button>
                )}
            </div>

            {/* Image preview */}
            {filePreview && fileType === 'image' && (
                <div className="rounded-xl border border-neutral-700 bg-neutral-800 p-4 flex justify-center">
                    <img src={filePreview} alt="Preview" className="max-h-64 object-contain rounded-md" />
                </div>
            )}

            {/* Text input — not disabled during processing so focus is never lost */}
            {!file && (
                <div className="rounded-xl border border-neutral-700 bg-neutral-800 overflow-hidden">
                    <Textarea
                        placeholder="Enter some text to count tokens…"
                        rows={10}
                        className="font-mono bg-transparent border-0 focus-visible:ring-0 resize-none p-4"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />
                </div>
            )}

            {error && <p className="text-orange-400 mb-2">{error}</p>}

            {selectedModel === OPUS_47_ID && (
                <div className="rounded-md border border-amber-600/40 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
                    <span className="font-medium">Heads up:</span> Claude Opus 4.7 uses a
                    new tokenizer that typically produces <em>more</em> tokens for the same
                    input than Opus 4.6 and earlier models. For reference we also run the
                    Opus 4.6 tokenizer and show both counts below.
                </div>
            )}

            <TokenMetrics
                tokens={stats.tokens ?? 0}
                gpt4oTokens={stats.gpt4oTokens}
                comparisonTokens={stats.comparisonTokens}
                comparisonModelName={stats.comparisonModel === OPUS_46_ID ? OPUS_46_NAME : stats.comparisonModel}
                chars={stats.chars}
                isProcessing={isProcessing}
                fileName={stats.fileName}
                fileType={file ? fileType : 'text'}
                model={selectedModelName}
                modelInputPricePerMTok={selectedModelPrice}
                comparisonInputPricePerMTok={comparisonModelPrice}
            />
        </div>
    );
};

interface TokenMetricsProps {
    tokens: number;
    gpt4oTokens: number | null;
    comparisonTokens: number | null;
    comparisonModelName: string | null;
    chars: number;
    isProcessing: boolean;
    fileName?: string;
    fileType: 'image' | 'pdf' | 'text' | 'unknown';
    model: string;
    modelInputPricePerMTok: number | null;
    comparisonInputPricePerMTok: number | null;
}

export const TokenMetrics = ({
    tokens, gpt4oTokens, comparisonTokens, comparisonModelName,
    chars, isProcessing, fileName, fileType, model,
    modelInputPricePerMTok, comparisonInputPricePerMTok,
}: TokenMetricsProps) => {
    const calcDiff = (compare: number | null, base: number): string => {
        if (compare === null || base === 0) return '';
        const diff = ((compare - base) / base) * 100;
        return diff < 0 ? ` (−${Math.abs(diff).toFixed(1)}%)` : ` (+${diff.toFixed(1)}%)`;
    };

    const gpt4oDiff = tokens > 0 && gpt4oTokens !== null ? calcDiff(gpt4oTokens, tokens) : '';
    const comparisonDiff = tokens > 0 && comparisonTokens !== null ? calcDiff(comparisonTokens, tokens) : '';

    return (
        <div className="flex flex-wrap gap-6 p-4 rounded-xl bg-neutral-800 border border-neutral-700">
            <div className="space-y-1">
                <h2 className="text-xs font-medium text-neutral-400">Claude Tokens</h2>
                <p className="text-3xl font-light">
                    {isProcessing ? <span className="animate-pulse">…</span> : tokens.toLocaleString()}
                </p>
                {!isProcessing && tokens > 0 && modelInputPricePerMTok !== null && (
                    <p className="text-xs text-neutral-500">
                        Est. input cost: {formatCost(tokens, modelInputPricePerMTok)}
                        <span className="text-neutral-600"> @ ${modelInputPricePerMTok}/MTok</span>
                    </p>
                )}
            </div>

            {comparisonModelName && (
                <div className="space-y-1">
                    <h2 className="text-xs font-medium text-neutral-400">{comparisonModelName} Tokens</h2>
                    <div className="flex items-baseline gap-2">
                        <p className="text-3xl font-light">
                            {isProcessing ? <span className="animate-pulse">…</span> : comparisonTokens !== null ? comparisonTokens.toLocaleString() : '—'}
                        </p>
                        {!isProcessing && comparisonTokens !== null && tokens > 0 && (
                            <span className={`text-sm ${comparisonDiff.includes('−') ? 'text-green-400' : 'text-orange-400'}`}>{comparisonDiff}</span>
                        )}
                    </div>
                    {!isProcessing && comparisonTokens !== null && comparisonTokens > 0 && comparisonInputPricePerMTok !== null && (
                        <p className="text-xs text-neutral-500">
                            Est. input cost: {formatCost(comparisonTokens, comparisonInputPricePerMTok)}
                            <span className="text-neutral-600"> @ ${comparisonInputPricePerMTok}/MTok</span>
                        </p>
                    )}
                </div>
            )}

            {(fileType === 'text' || !fileName) && (
                <div className="space-y-1">
                    <h2 className="text-xs font-medium text-neutral-400">GPT-4o Tokens</h2>
                    <div className="flex items-baseline gap-2">
                        <p className="text-3xl font-light">
                            {isProcessing ? <span className="animate-pulse">…</span> : gpt4oTokens !== null ? gpt4oTokens.toLocaleString() : '—'}
                        </p>
                        {!isProcessing && gpt4oTokens !== null && tokens > 0 && (
                            <span className={`text-sm ${gpt4oDiff.includes('−') ? 'text-green-400' : 'text-orange-400'}`}>{gpt4oDiff}</span>
                        )}
                    </div>
                    {!isProcessing && gpt4oTokens !== null && gpt4oTokens > 0 && (
                        <p className="text-xs text-neutral-500">
                            Est. input cost: {formatCost(gpt4oTokens, GPT4O_INPUT_PRICE_PER_MTOK)}
                            <span className="text-neutral-600"> @ ${GPT4O_INPUT_PRICE_PER_MTOK}/MTok</span>
                        </p>
                    )}
                </div>
            )}

            <div className="space-y-1">
                <h2 className="text-xs font-medium text-neutral-400">Characters</h2>
                <p className="text-3xl font-light">
                    {isProcessing ? <span className="animate-pulse">…</span> : chars.toLocaleString()}
                </p>
            </div>

            <div className="space-y-1">
                <h2 className="text-xs font-medium text-neutral-400">Model</h2>
                <p className="text-sm">{model}</p>
            </div>

            {fileName && (
                <div className="space-y-1 flex-1">
                    <h2 className="text-xs font-medium text-neutral-400">
                        {fileType === 'image' ? 'Image' : fileType === 'pdf' ? 'PDF' : 'File'}
                    </h2>
                    <p className="text-sm truncate">{fileName}</p>
                </div>
            )}
        </div>
    );
};

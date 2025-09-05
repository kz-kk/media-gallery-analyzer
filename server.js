const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const { exec, spawn } = require('child_process');

// --- In-memory progress (SSE) ---
const progressClients = new Map(); // key: normalized path, value: Set<res>
const progressState = new Map();   // key: normalized path, value: { featuresPct, whisperPct, message, done }

function normKey(p) {
  try { return String(p || '').replace(/\\/g, '/'); } catch { return String(p || ''); }
}
function pushProgress(filePath, payload) {
  const key = normKey(filePath);
  const prev = progressState.get(key) || {};
  const next = { ...prev, ...payload };
  progressState.set(key, next);
  const set = progressClients.get(key);
  if (set && set.size) {
    const data = `data: ${JSON.stringify(next)}\n\n`;
    for (const res of set) {
      try { res.write(data); } catch (_) {}
    }
  }
}

// シンプルかつクロスプラットフォームなパス検証
// - Windows でのドライブレター差異（C:\ vs c:\）や先頭スラッシュの扱いを考慮
// - baseDir を絶対パス化し、relative が '..' で始まらないことを確認
function isPathSafe(userPath, baseDir) {
    // baseDir を絶対・正規化
    const baseAbs = path.resolve(baseDir);
    // 対象パスを base を基準に解決
    const resolved = path.resolve(baseAbs, userPath);
    // base からの相対パスを取得
    const rel = path.relative(baseAbs, resolved);
    // rel が空（=同じ）か、上位参照で始まらず、絶対化されていないことを確認
    if (!rel) return true;
    const up = '..' + path.sep;
    return !rel.startsWith('..') && !rel.startsWith(up) && !path.isAbsolute(rel);
}

// Windows/Unix 共通でパスを正規化（検索/表示用）
function normalizePath(p) {
    return String(p || '').replace(/\\/g, '/');
}

// audio_indexer_v2.py に合わせた media_id 算出（sha256の先頭8バイトを整数化）
function computeMediaIdFromAbsPath(absPath) {
    try {
        const hash = crypto.createHash('sha256').update(String(absPath), 'utf8').digest();
        // 先頭8バイトをビッグエンディアン整数に
        let id = 0n;
        for (let i = 0; i < 8; i++) {
            id = (id << 8n) + BigInt(hash[i]);
        }
        // Number に収まらない環境でも Meili は文字列IDを受け付ける
        const asNumber = Number(id);
        return Number.isSafeInteger(asNumber) ? asNumber : id.toString();
    } catch (e) {
        // フォールバック: 16hex を整数化
        const hex = crypto.createHash('sha256').update(String(absPath), 'utf8').digest('hex').slice(0, 16);
        return parseInt(hex, 16);
    }
}

function computeMediaIdRel16(relPath) {
    try {
        const hex = crypto.createHash('sha256').update(String(relPath), 'utf8').digest('hex').slice(0, 16);
        return parseInt(hex, 16);
    } catch (e) {
        return null;
    }
}

function computeMediaIdRel8(relPath) {
    try {
        const hex = crypto.createHash('sha256').update(String(relPath), 'utf8').digest('hex').slice(0, 8);
        return parseInt(hex, 16);
    } catch (e) {
        return null;
    }
}

// Meilisearch 登録フォールバック（Python側で失敗した/未登録時に使用）
async function indexAudioToMeilisearchFallback(absPath, caption, tags, model, extra = {}) {
    try {
        const meiliUrl = process.env.MEILI_URL || 'http://127.0.0.1:17700';
        const meiliIndex = process.env.MEILI_INDEX || 'media';
        const meiliApiKey = process.env.MEILI_MASTER_KEY || 'masterKey';
        const baseDir = process.env.SCAN_PATH || '';
        const normBase = normalizePath(baseDir);
        const normAbs = normalizePath(absPath);
        let rel = normAbs;
        if (normBase && (normAbs === normBase || normAbs.startsWith(normBase + '/'))) {
            rel = normAbs.substring(normBase.length + (normAbs === normBase ? 0 : 1));
        }
        // 保存は画像と同じ rel8 に統一
        const id = computeMediaIdRel8(rel);
        const url = `${meiliUrl}/indexes/${meiliIndex}/documents`;
        const doc = {
            id,
            path: rel,
            caption: caption || '',
            tags: Array.isArray(tags) ? tags : [],
            model: model || 'audio-indexer-v2',
            type: 'audio',
            indexed_at: Math.floor(Date.now() / 1000),
            ...extra
        };
        const resp = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${meiliApiKey}`
            },
            body: JSON.stringify([doc])
        });
        const ok = resp.status === 200 || resp.status === 201 || resp.status === 202;
        console.log(`[DEBUG] Meilisearch fallback indexing: status=${resp.status}, ok=${ok}`);
        return ok;
    } catch (e) {
        console.warn('[DEBUG] Meilisearch fallback indexing failed:', e && e.message ? e.message : e);
        return false;
    }
}

// メディアファイル拡張子チェック
function isMediaFile(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mediaExts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg',
        '.mp3', '.wav', '.ogg', '.flac', '.m4a',
        // GPU圧縮テクスチャ（model-viewer/three.jsが参照）
        '.ktx2', '.basis',
        // 3D models (binary GLB only for now)
        '.glb'];
    return mediaExts.includes(ext);
}

// .envファイルを読み込む（dotenvパッケージなしで実装）
function loadEnv() {
    try {
        const envFile = fs.readFileSync('.env', 'utf8');
        envFile.split('\n').forEach(line => {
            if (line && !line.startsWith('#')) {
                const [key, value] = line.split('=');
                if (key && value) {
                    process.env[key.trim()] = value.trim();
                }
            }
        });
    } catch (err) {
        console.log('No .env file found, using defaults');
    }
}
loadEnv();

// --- venv ---
const candidates = [
    process.env.PYTHON_VENV_PATH && path.join(__dirname, process.env.PYTHON_VENV_PATH),
    path.join(__dirname, '.venv', 'Scripts', 'python.exe'),
    path.join(__dirname, 'venv', 'Scripts', 'python.exe'),
    path.join(__dirname, 'venv', 'bin', 'python')
].filter(Boolean);

const VENV_PYTHON = candidates.find(p => fs.existsSync(p)) || candidates[0];
console.log('[PY] Using Python interpreter:', VENV_PYTHON);

// 画像解析のためのヘルパー関数
function analyzeImageFile(filePath) {
    return new Promise((resolve, reject) => {
        const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
        const imageScript = process.env.IMAGE_INDEXER_SCRIPT || 'image_indexer.py';
        const pythonScript = path.join(__dirname, analysisDir, imageScript);
        const venvPython = VENV_PYTHON;
        
        // Virtual環境のPythonを使用（spawnで安全に）
        const args = [pythonScript, '--image', filePath];
        
        const child = spawn(venvPython, args, { 
            stdio: 'pipe',
            env: { 
                ...process.env, 
                PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__'),
                TRANSFORMERS_NO_TF: '1',
                TF_CPP_MIN_LOG_LEVEL: '2',
                PYTHONIOENCODING: 'utf-8',
                PYTHONUNBUFFERED: '1'
            }
        });
        
        let stdout = '';
        let stderr = '';
        
        child.stdout.on('data', (data) => {
            const s = data.toString();
            stdout += s;
            console.log('[audio][stdout]', s);
        });
        
        child.stderr.on('data', (data) => {
            const s = data.toString();
            stderr += s;
            console.warn('[audio][stderr]', s);
        });
        
        child.on('close', (code) => {
            if (code !== 0) {
                console.error('Python script error:', stderr);
                reject({ error: 'Analysis failed', details: stderr });
                return;
            }
            resolve({ success: true, output: stdout });
        });
        
        child.on('error', (error) => {
            console.error('Failed to start Python script:', error);
            reject({ error: 'Failed to start analysis', details: error.message });
        });
    });
}

// 音声解析のためのヘルパー関数
function analyzeAudioFile(filePath) {
    return new Promise((resolve, reject) => {
        const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
        // v1 は不要との方針に合わせ、既定を v2 に切替
        const audioScript = process.env.AUDIO_INDEXER_SCRIPT || 'audio_indexer_v2.py';
        const pythonScript = path.join(__dirname, analysisDir, audioScript);
        const venvPython = VENV_PYTHON;
        
        // Virtual環境のPythonを使用（spawnで安全に）
        // スクリプト世代に応じて引数を切り替え（LLMは使用しない方針で --no-lm-studio を常に付与）
        const args = [pythonScript, '--audio', filePath];
        // v2 のみ追加フラグに対応
        if (/audio_indexer_v2\.py$/i.test(audioScript)) {
            // 明示的に埋め込みモデルを渡す（重いPlamo固定を回避可能）
            const embModel = process.env.EMB_MODEL || 'pfnet/plamo-embedding-1b';
            args.push('--embedding_model', embModel);
            // v2はWhisper引数に対応
            const noWhisper = (process.env.NO_WHISPER === '1' || String(process.env.NO_WHISPER).toLowerCase() === 'true');
            if (noWhisper) {
                args.push('--no-whisper');
            } else {
                // 既定を 'small' に設定、速度と精度のバランスが最適
                const whisperModel = process.env.WHISPER_MODEL || 'small';
                args.push('--whisper-model', whisperModel);
                // 部分文字起こし（長尺対策）
                if (process.env.WHISPER_MAX_SECONDS) {
                    args.push('--whisper-max-seconds', String(process.env.WHISPER_MAX_SECONDS));
                }
                if (process.env.WHISPER_OFFSET_SECONDS) {
                    args.push('--whisper-offset-seconds', String(process.env.WHISPER_OFFSET_SECONDS));
                }
            }
            // v2はLM Studioを使わない（フラグ不要）
        }
        // 動画のように一部のみ解析（短尺）: 環境変数で調整可能。
        // v1(audio_indexer.py)のみ対応。v2は未対応なので渡さない。
        const isV2 = /audio_indexer_v2\.py$/i.test(audioScript);
        if (!isV2) {
            const chunkSec = process.env.AUDIO_CHUNK_SEC || '20';
            const maxChunks = process.env.AUDIO_MAX_CHUNKS || '1';
            args.push('--chunk-sec', String(chunkSec));
            args.push('--max-chunks', String(maxChunks));
            // v1でもLM Studioを無効化（フォールバックで確実に処理を完了させる）
            args.push('--no-lm-studio');
        }
        // LM Studio 側の待ち時間を短くして早めにフォールバック（v1のみ対応）
        if (!isV2) {
            const lmTimeout = process.env.AUDIO_LM_TIMEOUT || '20';
            args.push('--timeout', String(parseInt(lmTimeout) || 20));
        }
        
        // ネットワーク制限時の安全策: EMBEDDINGモデルを軽量に強制したい場合 NO_HF_DOWNLOAD=1 を使用
        const childEnv = { 
            ...process.env, 
            PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__'),
            TRANSFORMERS_NO_TF: '1',
            TF_CPP_MIN_LOG_LEVEL: '2',
            // Whisperモデルサイズを環境変数で渡す
            WHISPER_MODEL: process.env.WHISPER_MODEL || 'small',
            // Hugging Faceのキャッシュディレクトリを明示的に指定
            HF_HOME: process.env.HF_HOME || path.join(require('os').homedir(), '.cache', 'huggingface'),
            // マルチプロセッシングの設定
            TOKENIZERS_PARALLELISM: 'false',
            PYTHONUNBUFFERED: '1'
        };
        if (process.env.NO_HF_DOWNLOAD === '1' && /audio_indexer_v2\.py$/i.test(audioScript)) {
            childEnv.EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
            childEnv.NO_WHISPER = childEnv.NO_WHISPER || '1';
        }

        console.log('[Audio] Executing command:', venvPython, args.join(' '));
        
        const child = spawn(venvPython, args, { stdio: 'pipe', env: { ...childEnv, PYTHONIOENCODING: 'utf-8' } });
        
        let stdout = '';
        let stderr = '';
        
        child.stdout.on('data', (data) => {
            const chunk = data.toString();
            stdout += chunk;
            console.log('[Audio][stdout]', chunk);
            // Parse progress and broadcast via SSE
            try {
                const lines = chunk.split(/\r?\n/);
                for (const line of lines) {
                    if (!line) continue;
                    if (line.includes('[A]') && line.includes('進捗')) {
                        const m = line.match(/([0-9]+(?:\.[0-9]+)?)%/);
                        if (m) pushProgress(filePath, { featuresPct: parseFloat(m[1]), message: 'features' });
                    }
                    if (line.includes('[B]') && line.includes('Whisper')) {
                        const m = line.match(/([0-9]+(?:\.[0-9]+)?)%/);
                        if (m) pushProgress(filePath, { whisperPct: parseFloat(m[1]), message: 'whisper' });
                    }
                    if (line.startsWith('AUDIO_ANALYSIS_RESULT: ')) {
                        pushProgress(filePath, { featuresPct: 100, whisperPct: 100, done: true });
                    }
                }
            } catch (e) {}
        });
        
        child.stderr.on('data', (data) => {
            const chunk = data.toString();
            stderr += chunk;
            console.log('[Audio][stderr]', chunk);
        });
        // タイムアウト監視（デフォルト3分、AUDIO_TIMEOUT_MS環境変数で変更可）
        const timeoutMs = parseInt(process.env.AUDIO_TIMEOUT_MS || '180000');
        const timer = setTimeout(() => {
            console.warn('[Audio] analysis timeout, killing process');
            try { child.kill('SIGKILL'); } catch (e) {}
        }, timeoutMs);

        child.on('close', (code) => {
            clearTimeout(timer);
            if (code !== 0) {
                console.error('Audio analysis script error:', stderr);
                pushProgress(filePath, { message: 'error', done: true });
                reject({ error: 'Audio analysis failed', details: stderr });
                return;
            }
            pushProgress(filePath, { featuresPct: 100, whisperPct: 100, done: true });
            resolve({ success: true, output: stdout });
        });
        
        child.on('error', (error) => {
            console.error('Failed to start audio analysis script:', error);
            reject({ error: 'Failed to start audio analysis', details: error.message });
        });
    });
}

// 動画解析のためのヘルパー関数
function analyzeVideoFile(filePath) {
    return new Promise((resolve, reject) => {
        const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
        const videoScript = process.env.VIDEO_INDEXER_SCRIPT || 'video_indexer.py';
        const pythonScript = path.join(__dirname, analysisDir, videoScript);
        const venvPython = VENV_PYTHON;
        
        // Virtual環境のPythonを使用（spawnで安全に）
        const args = [pythonScript, '--video', filePath, '--max_frames', '3'];
        
        const child = spawn(venvPython, args, { 
            stdio: 'pipe',
            env: { 
                ...process.env, 
                PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__'),
                TRANSFORMERS_NO_TF: '1',
                TF_CPP_MIN_LOG_LEVEL: '2',
                PYTHONWARNINGS: 'ignore:::tensorflow',
                PYTHONIOENCODING: 'utf-8',
                PYTHONUNBUFFERED: '1'
            }
        });
        
        let stdout = '';
        let stderr = '';
        
        child.stdout.on('data', (data) => {
            const s = data.toString();
            stdout += s;
            console.log('[video][stdout]', s);
        });
        
        child.stderr.on('data', (data) => {
            const s = data.toString();
            stderr += s;
            console.warn('[video][stderr]', s);
        });
        
        child.on('close', (code) => {
            if (code !== 0) {
                console.error('Video analysis script error:', stderr);
                reject({ error: 'Video analysis failed', details: stderr });
                return;
            }
            resolve({ success: true, output: stdout });
        });
        
        child.on('error', (error) => {
            console.error('Failed to start video analysis script:', error);
            reject({ error: 'Failed to start video analysis', details: error.message });
        });
    });
}

// Meilisearch検索関数
async function searchWithMeilisearch(query, limit = 20) {
    try {
        const meiliUrl = process.env.MEILI_URL || 'http://127.0.0.1:17700';
        const meiliIndex = process.env.MEILI_INDEX || 'media';
        const meiliApiKey = process.env.MEILI_MASTER_KEY || 'masterKey';
        
        const searchUrl = `${meiliUrl}/indexes/${meiliIndex}/search`;
            const searchBody = JSON.stringify({
            q: query,
            limit: parseInt(limit) || 20,
            attributesToRetrieve: ['id', 'path', 'caption', 'tags', 'model', 'category', 'instruments']
        });
        
        console.log(`[DEBUG] Meilisearch query: "${query}", limit: ${limit}`);
        
        const response = await fetch(searchUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${meiliApiKey}`
            },
            body: searchBody
        });
        
        if (!response.ok) {
            throw new Error(`Meilisearch API error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log(`[DEBUG] Meilisearch found ${result.hits.length} results`);
        
        // パスを相対パスに変換（Windows区切りも考慮）
        const basePath = process.env.SCAN_PATH;
        const normBase = normalizePath(basePath || '');
        let processedHits = result.hits.map(hit => {
            const orig = hit.path;
            const norm = normalizePath(orig);
            if (norm && normBase && (norm.startsWith(normBase + '/') || norm === normBase)) {
                hit.path = norm.substring(normBase.length + (norm === normBase ? 0 : 1));
            } else if (orig) {
                // 既に相対/別ドライブの可能性。スラッシュに正規化だけ適用
                hit.path = norm;
            }
            return hit;
        });
        // .snapshots 配下は除外
        processedHits = processedHits.filter(hit => {
            const p = String(hit.path || '');
            return !(p.includes('/.snapshots/') || p.includes('\\.snapshots\\'));
        });
        
        return processedHits;
    } catch (err) {
        console.error('Meilisearch search error:', err);
        // エラー時は空の配列を返す
        return [];
    }
}


// Qdrantセマンティック検索関数
async function searchWithQdrant(query, limit = 20) {
    try {
        const qdrantUrl = process.env.QDRANT_URL || 'http://127.0.0.1:26333';
        const baseCollection = process.env.QDRANT_COLLECTION || 'media_vectors';

        const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
        const embeddingScriptName = process.env.EMBEDDING_SCRIPT || 'get_embedding.py';
        const embeddingScript = path.join(__dirname, analysisDir, embeddingScriptName);
        const venvPython = VENV_PYTHON;

        async function embedWithModel(modelName) {
            return new Promise((resolve, reject) => {
                const args = [embeddingScript, query];
                const env = { 
                    ...process.env, 
                    EMB_MODEL: modelName, 
                    PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__'), 
                    PYTHONIOENCODING: 'utf-8',
                    PYTHONUNBUFFERED: '1'
                };
                const child = spawn(venvPython, args, { stdio: 'pipe', env });
                let stdout = '', stderr = '';
                child.stdout.on('data', d => {
                    const s = d.toString();
                    stdout += s;
                    console.log('[model][stdout]', s);
                    // Progress hook: lines like "MODEL_PROGRESS: <stage>" or "MODEL_PROGRESS: <pct>% <stage>"
                    try {
                        const lines = s.split(/\r?\n/);
                        for (const line of lines) {
                            if (!line) continue;
                            if (line.startsWith('MODEL_PROGRESS:')) {
                                const rest = line.substring('MODEL_PROGRESS:'.length).trim();
                                // Try to parse percentage
                                const m = rest.match(/(\d+(?:\.\d+)?)%/);
                                const pct = m ? parseFloat(m[1]) : undefined;
                                pushProgress(filePath, { message: rest, ...(pct !== undefined ? { featuresPct: pct } : {}) });
                            }
                            if (line.startsWith('MODEL_ANALYSIS_RESULT: ')) {
                                pushProgress(filePath, { message: 'done', done: true });
                            }
                        }
                    } catch (_) {}
                });
                child.stderr.on('data', d => { const s = d.toString(); stderr += s; console.warn('[model][stderr]', s); });
                child.on('close', code => {
                    if (code !== 0) return reject(new Error(stderr));
                    try {
                        const obj = JSON.parse(stdout);
                        console.log(`[DEBUG] Embedding model: ${modelName}, dim: ${Array.isArray(obj.embedding) ? obj.embedding.length : 'n/a'}`);
                        resolve(obj.embedding);
                    } catch (e) { reject(new Error('Invalid embedding response')); }
                });
                child.on('error', err => reject(err));
            });
        }

        async function qdrantSearch(collection, embedding) {
            const searchUrl = `${qdrantUrl}/collections/${collection}/points/search`;
            const searchBody = {
                vector: embedding,
                limit: Math.max(50, (parseInt(limit) || 20) * 3),
                with_payload: true,
                params: { hnsw_ef: Math.max(128, (parseInt(limit) || 20) * 4), exact: false }
            };
            console.log(`[DEBUG] Qdrant request URL: ${searchUrl}`);
            const response = await fetch(searchUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(searchBody) });
            if (!response.ok) {
                const errorText = await response.text();
                console.error(`[DEBUG] Qdrant API error details: ${response.status} - ${errorText}`);
                return [];
            }
            const result = await response.json();
            const basePath = process.env.SCAN_PATH;
            const SIMILARITY_THRESHOLD = parseFloat(process.env.QDRANT_THRESHOLD || '0.25');
            const raw = (result.result || []);
            console.log(`[DEBUG] Qdrant raw results: ${raw.length}`);
            let items = raw
                .filter(hit => hit.score >= SIMILARITY_THRESHOLD)
                .map(hit => {
                    const payload = hit.payload;
                    let displayPath = payload.path;
                    if (displayPath && basePath && displayPath.startsWith(basePath)) {
                        displayPath = displayPath.substring(basePath.length + 1);
                    }
                    return { id: payload.media_id, path: displayPath, caption: payload.caption, tags: payload.tags, model: payload.model, category: payload.category, instruments: payload.instruments, score: hit.score };
                });
            // .snapshots 配下は除外
            items = items.filter(it => !(it.path && it.path.includes('/.snapshots/')));
            console.log(`[DEBUG] Qdrant mapped results (>=${SIMILARITY_THRESHOLD}): ${items.length}`);
            if (items.length === 0 && raw.length > 0) {
                // 閾値で0件のときは上位をフォールバックで返す
                console.log('[DEBUG] No results above threshold; falling back to top results');
                items = raw
                    .sort((a,b) => b.score - a.score)
                    .slice(0, Math.min(10, raw.length))
                    .map(hit => {
                        const payload = hit.payload;
                        let displayPath = payload.path;
                        if (displayPath && basePath && displayPath.startsWith(basePath)) {
                            displayPath = displayPath.substring(basePath.length + 1);
                        }
                        return { id: payload.media_id, path: displayPath, caption: payload.caption, tags: payload.tags, model: payload.model, category: payload.category, instruments: payload.instruments, score: hit.score };
                    });
                items = items.filter(it => !(it.path && it.path.includes('/.snapshots/')));
            }
            // スコア順にソート（降順）
            items.sort((a,b) => b.score - a.score);
            return items;
        }

        console.log(`[DEBUG] Qdrant semantic search: "${query}", limit: ${limit}`);

        // Plamoのみ使用
        const primaryModel = 'pfnet/plamo-embedding-1b';
        const primaryCollection = baseCollection;

        // Run Plamo search only
        const primaryEmbedding = await embedWithModel(primaryModel);
        console.log(`[DEBUG] Using collection: ${primaryCollection}`);
        let results = await qdrantSearch(primaryCollection, primaryEmbedding);

        // Rerank if enough
        const minRerank = parseInt(process.env.RERANK_MIN_RESULTS || '1');
        if (results.length >= minRerank) {
            const topK = Math.max(minRerank, parseInt(process.env.RERANK_TOP || '50'));
            const toRerank = results.slice(0, topK);
            try {
                const reranked = await rerankResults(query, toRerank);
                // 以降の残りは元の順序のまま後ろに付与
                const rest = results.slice(topK);
                return reranked.concat(rest);
            } catch (e) {
                console.warn('[DEBUG] Rerank skipped due to error:', e && e.message ? e.message : e);
                return results;
            }
        }
        return results;
    } catch (err) {
        console.error('Qdrant search error:', err);
        return [];
    }
}

// Rerank機能
async function rerankResults(query, results) {
    return new Promise((resolve, reject) => {
        const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
        const rerankScriptName = process.env.RERANK_SCRIPT || 'rerank_results.py';
        const rerankScript = path.join(__dirname, analysisDir, rerankScriptName);
        const venvPython = VENV_PYTHON;

        const resultsJson = JSON.stringify(results);
        console.log(`[DEBUG] Starting rerank (spawn) for query: "${query}", script: ${rerankScript}`);

        const child = spawn(venvPython, [rerankScript, query, resultsJson], {
            stdio: 'pipe',
            env: { ...process.env, PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__'), PYTHONIOENCODING: 'utf-8' }
        });

        let stdout = '';
        let stderr = '';
        child.stdout.on('data', d => stdout += d.toString());
        child.stderr.on('data', d => { const s = d.toString(); stderr += s; console.log('[RERANK DEBUG]', s); });
        child.on('close', code => {
            if (code !== 0) {
                console.error('Rerank script exited with code:', code);
                return reject(new Error(stderr || `rerank exit ${code}`));
            }
            try {
                const result = JSON.parse(stdout);
                if (result.success && Array.isArray(result.results)) {
                    console.log(`[DEBUG] Rerank completed, ${result.results.length} results reordered`);
                    resolve(result.results);
                } else {
                    console.error('Rerank failed (no success):', result && result.error);
                    reject(new Error((result && result.error) || 'Rerank failed'));
                }
            } catch (e) {
                console.error('Failed to parse rerank JSON:', e);
                console.error('Raw stdout:', stdout);
                reject(e);
            }
        });
        child.on('error', err => {
            console.error('Failed to start rerank process:', err);
            reject(err);
        });
    });
}

// 検索関数（typeによって検索方法を選択）
async function searchMedia(query, type = 'text', limit = 20) {
    try {
        console.log(`[DEBUG] Search query: "${query}", type: ${type}, limit: ${limit}`);
        
        if (type === 'text') {
            // テキスト検索はMeilisearchを使用
            return await searchWithMeilisearch(query, limit);
        } else if (type === 'fulltext') {
            // Meilisearch全文検索
            return await searchWithMeilisearch(query, limit);
        } else if (type === 'semantic') {
            // Qdrantセマンティック検索
            return await searchWithQdrant(query, limit);
        } else {
            return [];
        }
    } catch (err) {
        console.error('Search error:', err);
        throw err;
    }
}

async function getMediaById(id) {
    // TODO: Implement with Meilisearch or Qdrant
    console.warn('getMediaById not implemented after MySQL removal');
    return null;
}

// メディアファイルの拡張子
const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp'];
const videoExtensions = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
const audioExtensions = ['mp3', 'wav', 'ogg', 'flac', 'm4a'];
const modelExtensions = ['glb'];

// ディレクトリをスキャンする関数
function scanDirectory(dirPath, baseDir = null, depth = 0, maxDepth = null) {
    // デフォルト値は環境変数から取得、未設定時は5
    if (maxDepth === null) {
        maxDepth = parseInt(process.env.MAX_DEPTH) || 5;
    }
    if (depth > maxDepth) return [];
    
    if (!baseDir) baseDir = dirPath;
    
    const result = {
        files: [],
        folders: []
    };
    
    try {
        const items = fs.readdirSync(dirPath);
        
        items.forEach(item => {
            // 隠しファイルやシステムファイルをスキップ
            if (item.startsWith('.') || item === 'node_modules') return;
            
            const fullPath = path.join(dirPath, item);
            const relativePath = path.relative(baseDir, fullPath).replace(/\\/g, '/');
            const stat = fs.statSync(fullPath);
            
            if (stat.isDirectory()) {
                const subItems = scanDirectory(fullPath, baseDir, depth + 1, maxDepth);
                result.folders.push({
                    name: item,
                    path: relativePath,
                    items: subItems
                });
            } else if (stat.isFile()) {
                const ext = path.extname(item).toLowerCase().slice(1);
                let type = 'other';
                
                if (imageExtensions.includes(ext)) type = 'image';
                else if (videoExtensions.includes(ext)) type = 'video';
                else if (audioExtensions.includes(ext)) type = 'audio';
                else if (modelExtensions.includes(ext)) type = 'model';
                
                if (type !== 'other') {
                    result.files.push({
                        name: item,
                        path: relativePath,
                        fullPath: fullPath,
                        type: type,
                        size: stat.size,
                        modified: stat.mtime
                    });
                }
            }
        });
    } catch (err) {
        console.error('Error scanning directory:', dirPath, err);
    }
    
    return result;
}

// HTTPサーバーの作成
const server = http.createServer(async (req, res) => {
    // CORS設定
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    // セキュリティヘッダー
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    // CSPヘッダー設定（model-viewer/wasm/Blob URL対応）
    // - wasm: 'wasm-unsafe-eval' 許可
    // - blob: Blob URLのfetch許可（GLTF/テクスチャの中間URL）
    // - unpkg: model-viewerの追加取得を許可
    res.setHeader('Content-Security-Policy',
        "default-src 'self'; " +
        "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; " +
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
        "font-src 'self' https://fonts.gstatic.com data:; " +
        "img-src 'self' data: blob:; " +
        "media-src 'self'; " +
        "connect-src 'self' http://127.0.0.1:* http://localhost:* blob: data:; " +
        "worker-src 'self' blob:; child-src 'self' blob:"
    );
    
    // OPTIONS リクエストへの対応（CORS preflight）
    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }
    
    // 設定情報のエンドポイント
    if (req.url === '/api/config') {
        if (!process.env.PORT || !process.env.SCAN_PATH) {
            res.setHeader('Content-Type', 'application/json');
            res.writeHead(500);
            res.end(JSON.stringify({
                error: 'Missing required environment variables: PORT and/or SCAN_PATH'
            }));
            return;
        }
        res.setHeader('Content-Type', 'application/json');
        res.writeHead(200);
        res.end(JSON.stringify({
            port: process.env.PORT,
            scanPath: process.env.SCAN_PATH,
            meilisearchUrl: process.env.MEILI_URL || `http://${process.env.SERVER_HOST || '127.0.0.1'}:17700`,
            serverBaseUrl: `http://${process.env.SERVER_HOST || '127.0.0.1'}:${process.env.PORT}`
        }));
        return;
    }
    
    // .snapshots の存在チェック用HEAD（404を出さずに静かに判定）
    if (req.method === 'HEAD' && req.url.startsWith('/.snapshots/')) {
        try {
            const baseDir = process.env.SCAN_PATH;
            if (!baseDir) {
                res.writeHead(500);
                res.end();
                return;
            }
            // URLのパース（クエリを除外）
            const parsed = new URL(req.url, `http://${process.env.SERVER_HOST || '127.0.0.1'}:${process.env.PORT || 80}`);
            const filePath = decodeURIComponent(parsed.pathname.substring(1)); // remove leading '/'
            if (!isPathSafe(filePath, baseDir)) {
                res.writeHead(403);
                res.end();
                return;
            }
            const fullPath = path.join(baseDir, filePath);
            if (fs.existsSync(fullPath)) {
                // 存在する: 200（コンソールにエラーを出さない）
                res.writeHead(200);
            } else {
                // 存在しない: 204 No Content（404だとブラウザがエラー表示するため）
                res.writeHead(204);
            }
            res.end();
        } catch (e) {
            res.writeHead(204);
            res.end();
        }
        return;
    }

    // 静的ファイルの配信（画像、動画、音声）
    if (req.method === 'GET' && !req.url.startsWith('/api/')) {
        // URLを一度パースしてクエリを除いたパス名で判定・解決
        const parsed = new URL(req.url, `http://${process.env.SERVER_HOST || '127.0.0.1'}:${process.env.PORT || 80}`);
        const pathname = parsed.pathname;
        // index.htmlの配信
        if (pathname === '/' || pathname === '/index.html') {
            const indexPath = path.join(__dirname, process.env.INDEX_HTML_PATH || 'index.html');
            fs.readFile(indexPath, (err, data) => {
                if (err) {
                    res.writeHead(404);
                    res.end('Not found');
                    return;
                }
                res.setHeader('Content-Type', 'text/html');
                res.writeHead(200);
                res.end(data);
            });
            return;
        }

        // favicon（存在しないため204で抑止）
        if (pathname === '/favicon.ico') {
            res.writeHead(204);
            res.end();
            return;
        }

        // シンプルなGLBビューワ（CDN不使用、model-viewerローカル）
        if (pathname === '/glb-viewer.html') {
            const viewerPath = path.join(__dirname, 'glb-viewer.html');
            fs.readFile(viewerPath, (err, data) => {
                if (err) {
                    res.writeHead(404);
                    res.end('Not found');
                    return;
                }
                res.setHeader('Content-Type', 'text/html');
                res.writeHead(200);
                res.end(data);
            });
            return;
        }
        
        // UI用の静的アセット配信（JS/CSS/フォントのみに限定）
        if (pathname.startsWith('/assets/js/') || pathname.startsWith('/assets/css/') || pathname.startsWith('/assets/fonts/')) {
            const assetPath = path.join(__dirname, pathname);
            fs.readFile(assetPath, (err, data) => {
                if (err) { res.writeHead(404); res.end('Not found'); return; }
                const ext = path.extname(assetPath).toLowerCase();
                const mimeTypes = {
                    '.js': 'application/javascript',
                    '.css': 'text/css',
                    '.map': 'application/json',
                    '.woff': 'font/woff',
                    '.woff2': 'font/woff2'
                };
                const contentType = mimeTypes[ext] || 'application/octet-stream';
                res.setHeader('Content-Type', contentType);
                res.writeHead(200);
                res.end(data);
            });
            return;
        }

        // model-viewer が動的に参照するデコーダーファイルの別名提供（ルート直下参照対策）
        const decoderAliases = new Map([
            // Meshopt (non-module UMD build expected by model-viewer)
            ['/meshopt_decoder.js', path.join(__dirname, 'assets/js/vendor/meshopt_decoder.js')],
            ['/meshopt_decoder.wasm', path.join(__dirname, 'assets/js/vendor/meshopt_decoder.wasm')],
            // Keep module aliases in case some path tries them
            ['/meshopt_decoder.module.js', path.join(__dirname, 'assets/js/vendor/meshopt_decoder.module.js')],
            ['/meshopt_decoder.module.wasm', path.join(__dirname, 'assets/js/vendor/meshopt_decoder.module.wasm')],
            ['/draco_wasm_wrapper.js', path.join(__dirname, 'assets/js/vendor/draco_wasm_wrapper.js')],
            // Some loaders expect this exact name
            ['/draco_decoder.js', path.join(__dirname, 'assets/js/vendor/draco_wasm_wrapper.js')],
            ['/draco_decoder.wasm', path.join(__dirname, 'assets/js/vendor/draco_decoder.wasm')]
        ]);
        if (decoderAliases.has(pathname)) {
            const p = decoderAliases.get(pathname);
            fs.readFile(p, (err, data) => {
                if (err) {
                    res.writeHead(404);
                    res.end('Not found');
                    return;
                }
                const ext = path.extname(p).toLowerCase();
                const mime = ext === '.js' ? 'application/javascript'
                    : (ext === '.wasm' ? 'application/wasm' : 'application/octet-stream');
                res.setHeader('Content-Type', mime);
                res.writeHead(200);
                res.end(data);
            });
            return;
        }
        
        // メディアファイルの配信
        if (!process.env.SCAN_PATH) {
            res.writeHead(500);
            res.end('ERROR: SCAN_PATH environment variable is not set');
            return;
        }
        const baseDir = process.env.SCAN_PATH;
        const filePath = decodeURIComponent(pathname.substring(1)); // 先頭の/を削除（クエリを除外）
        
        console.log(`[DEBUG] Request for: ${filePath}, baseDir: ${baseDir}`);
        
        // パストラバーサル攻撃をチェック
        if (!isPathSafe(filePath, baseDir)) {
            console.warn(`[SECURITY] Path traversal attempt: ${filePath}`);
            res.writeHead(403);
            res.end('Access forbidden');
            return;
        }
        
        const fullPath = path.join(baseDir, filePath);
        
        // メディアファイルのみ許可
        if (!isMediaFile(fullPath)) {
            res.writeHead(403);
            res.end('Access forbidden: Media files only');
            return;
        }
        
        // ファイルの存在確認
        fs.stat(fullPath, (err, stats) => {
            if (err || !stats.isFile()) {
                res.writeHead(404);
                res.end('File not found');
                return;
            }
            
            // MIMEタイプの設定
            const ext = path.extname(fullPath).toLowerCase();
            const mimeTypes = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.webp': 'image/webp',
                '.mp4': 'video/mp4',
                '.mov': 'video/quicktime',
                '.avi': 'video/x-msvideo',
                '.mkv': 'video/x-matroska',
                '.webm': 'video/webm',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.ogg': 'audio/ogg',
                '.flac': 'audio/flac',
                '.m4a': 'audio/mp4',
                // GPU圧縮テクスチャ
                '.ktx2': 'application/octet-stream',
                '.basis': 'application/octet-stream',
                // 3D
                '.glb': 'model/gltf-binary'
            };
            
            const contentType = mimeTypes[ext] || 'application/octet-stream';
            res.setHeader('Content-Type', contentType);
            res.setHeader('Cache-Control', 'public, max-age=3600');
            
            // ファイルをストリーミング
            const stream = fs.createReadStream(fullPath);
            stream.pipe(res);
            stream.on('error', () => {
                res.writeHead(500);
                res.end('Internal server error');
            });
        });
        return;
    }
    
    res.setHeader('Content-Type', 'application/json');
    
    // 画像解析APIエンドポイント（既存ファイル用）
    if (req.url === '/api/analyze-image' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', async () => {
            try {
                const { filePath } = JSON.parse(body);
                if (!filePath) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'File path is required' }));
                    return;
                }
                
                // パストラバーサル攻撃をチェック
                const baseDir = process.env.SCAN_PATH;
                if (!isPathSafe(filePath, baseDir)) {
                    console.warn(`[SECURITY] Path traversal in analyze-image: ${filePath}`);
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Invalid file path' }));
                    return;
                }
                
                const fullPath = path.join(baseDir, filePath);
                
                // ファイルの存在確認
                if (!fs.existsSync(fullPath)) {
                    res.writeHead(404);
                    res.end(JSON.stringify({ error: 'File not found' }));
                    return;
                }
                
                // 画像解析実行
                try {
                    const result = await analyzeImageFile(fullPath);
                    
                    // Pythonスクリプトの出力から解析結果を抽出
                    let analysisResult = null;
                    if (result.output) {
                        const lines = result.output.split('\n');
                        for (const raw of lines) {
                            const idx = raw.indexOf('RESULT_JSON:');
                            if (idx !== -1) {
                                try {
                                    analysisResult = JSON.parse(raw.slice(idx + 'RESULT_JSON:'.length));
                                    break;
                                } catch (e) {
                                    console.error('Failed to parse result JSON:', e);
                                }
                            }
                        }
                    }
                    
                    console.log(`[DEBUG] Analysis result:`, analysisResult);
                    
                    if (analysisResult && analysisResult.success) {
                        res.writeHead(200);
                        res.end(JSON.stringify({
                            success: true,
                            message: 'Image analyzed and indexed successfully',
                            result: analysisResult
                        }));
                    } else {
                        res.writeHead(200);
                        res.end(JSON.stringify({
                            success: true,
                            message: 'Image analyzed but result extraction failed',
                            result: analysisResult
                        }));
                    }
                    
                } catch (analysisError) {
                    if (!res.headersSent) {
                        res.writeHead(500);
                        res.end(JSON.stringify({ 
                            error: 'Analysis failed', 
                            details: analysisError.details || analysisError.message || String(analysisError)
                        }));
                    }
                }
                
            } catch (err) {
                console.error('Parse request error:', err);
                if (!res.headersSent) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Invalid JSON request' }));
                }
            }
        });
        return;
    }
    
    // 動画解析APIエンドポイント
    if (req.url === '/api/analyze-video' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', async () => {
            try {
                const { filePath } = JSON.parse(body);
                if (!filePath) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'File path is required' }));
                    return;
                }
                
                // パストラバーサル攻撃をチェック
                const baseDir = process.env.SCAN_PATH;
                if (!isPathSafe(filePath, baseDir)) {
                    console.warn(`[SECURITY] Path traversal in analyze-video: ${filePath}`);
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Invalid file path' }));
                    return;
                }
                
                const fullPath = path.join(baseDir, filePath);
                
                // ファイルの存在確認
                if (!fs.existsSync(fullPath)) {
                    res.writeHead(404);
                    res.end(JSON.stringify({ error: 'File not found' }));
                    return;
                }
                
                // 動画解析実行
                try {
                    const result = await analyzeVideoFile(fullPath);
                    
                    // Pythonスクリプトの出力から構造化されたJSONを抽出
                    let analysisResult = null;
                    if (result.output) {
                        const lines = result.output.split('\n');
                        for (const line of lines) {
                            if (line.startsWith('VIDEO_ANALYSIS_RESULT: ')) {
                                try {
                                    const jsonStr = line.substring('VIDEO_ANALYSIS_RESULT: '.length);
                                    analysisResult = JSON.parse(jsonStr);
                                    break;
                                } catch (parseError) {
                                    console.error('Failed to parse video analysis result JSON:', parseError);
                                }
                            }
                        }
                        
                        // 古い形式のフォールバック（後方互換性）
                        if (!analysisResult) {
                            for (const line of lines) {
                                if (line.includes('✅ 登録完了:')) {
                                    analysisResult = {
                                        success: true,
                                        message: line.trim(),
                                        file_name: line.split('✅ 登録完了: ')[1]
                                    };
                                    break;
                                } else if (line.includes('❌ エラー:')) {
                                    analysisResult = {
                                        success: false,
                                        message: line.trim()
                                    };
                                    break;
                                }
                            }
                        }
                    }
                    
                    console.log(`[DEBUG] Video analysis result:`, analysisResult);
                    
                    if (analysisResult && analysisResult.success) {
                        res.writeHead(200);
                        res.end(JSON.stringify({
                            success: true,
                            message: 'Video analyzed and indexed successfully',
                            result: analysisResult
                        }));
                    } else {
                        res.writeHead(200);
                        res.end(JSON.stringify({
                            success: true,
                            message: 'Video analyzed but result extraction failed',
                            result: analysisResult
                        }));
                    }
                    
                } catch (analysisError) {
                    if (!res.headersSent) {
                        res.writeHead(500);
                        res.end(JSON.stringify({ 
                            error: 'Video analysis failed', 
                            details: analysisError.details || analysisError.message || String(analysisError)
                        }));
                    }
                }
                
            } catch (err) {
                console.error('Parse request error:', err);
                if (!res.headersSent) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Invalid JSON request' }));
                }
            }
        });
        return;
    }
    
    // 音声解析APIエンドポイント
  if (req.url === '/api/analyze-audio' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', async () => {
            try {
                const { filePath } = JSON.parse(body);
                console.log('[AUDIO] POST /api/analyze-audio', { filePath });
                if (!filePath) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'File path is required' }));
                    return;
                }
                
                // パストラバーサル攻撃をチェック
                const baseDir = process.env.SCAN_PATH;
                if (!isPathSafe(filePath, baseDir)) {
                    console.warn(`[SECURITY] Path traversal in analyze-audio: ${filePath}`);
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Invalid file path' }));
                    return;
                }
                
                const fullPath = path.join(baseDir, filePath);
                console.log('[AUDIO] resolved fullPath', fullPath);
                
                // ファイルの存在確認
                if (!fs.existsSync(fullPath)) {
                    console.warn('[AUDIO] File not found', fullPath);
                    res.writeHead(404);
                    res.end(JSON.stringify({ error: 'File not found' }));
                    return;
                }
                
                // 音声解析実行
                try {
                    const result = await analyzeAudioFile(fullPath);
                    console.log('[AUDIO] analyzeAudioFile completed');
                    
                    // Pythonスクリプトの出力から構造化されたJSONを抽出
                    let analysisResult = null;
                    if (result.output) {
                        const lines = result.output.split('\n');
                        for (const line of lines) {
                            if (line.startsWith('AUDIO_ANALYSIS_RESULT: ')) {
                                try {
                                    const jsonStr = line.substring('AUDIO_ANALYSIS_RESULT: '.length);
                                    analysisResult = JSON.parse(jsonStr);
                                    break;
                                } catch (parseError) {
                                    console.error('Failed to parse audio analysis result JSON:', parseError);
                                }
                            }
                        }
                    }
                    
                    console.log(`[DEBUG] Audio analysis result:`, analysisResult);
                    
                    if (analysisResult && analysisResult.success) {
                        // Python側のMeili登録が失敗した場合に備えたフォールバック登録
                        try {
                            const extra = {};
                            if (typeof analysisResult.duration === 'number') extra.duration = analysisResult.duration;
                            await indexAudioToMeilisearchFallback(fullPath, analysisResult.caption, analysisResult.tags, analysisResult.model, extra);
                        } catch (e) {
                            console.warn('[AUDIO] Fallback indexing skipped due to error:', e && e.message ? e.message : e);
                        }
                        res.writeHead(200);
                        res.end(JSON.stringify({
                            success: true,
                            message: 'Audio analyzed and indexed successfully',
                            result: analysisResult
                        }));
                    } else {
                        res.writeHead(200);
                        res.end(JSON.stringify({
                            success: true,
                            message: 'Audio analyzed but result extraction failed',
                            result: analysisResult
                        }));
                    }
                    
                } catch (analysisError) {
                    console.error('[AUDIO] analyzeAudioFile error', analysisError);
                    if (!res.headersSent) {
                        res.writeHead(500);
                        res.end(JSON.stringify({ 
                            error: 'Audio analysis failed', 
                            details: analysisError.details || analysisError.message || String(analysisError)
                        }));
                    }
                }
                
            } catch (err) {
                console.error('Parse request error:', err);
                if (!res.headersSent) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Invalid JSON request' }));
                }
            }
        });
        return;
    }

    // SSE progress endpoint
    if (req.url.startsWith('/api/progress') && req.method === 'GET') {
        try {
            const urlObj = new URL(req.url, `http://${req.headers.host}`);
            const filePath = urlObj.searchParams.get('path');
            if (!filePath) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'path query is required' }));
                return;
            }
            const key = normKey(filePath);
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            });
            res.write('\n');
            // Register client
            let set = progressClients.get(key);
            if (!set) { set = new Set(); progressClients.set(key, set); }
            set.add(res);
            // Send last state if exists
            const state = progressState.get(key);
            if (state) res.write(`data: ${JSON.stringify(state)}\n\n`);
            // Heartbeat
            const hb = setInterval(() => { try { res.write(': ping\n\n'); } catch (_) {} }, 15000);
            // Cleanup
            req.on('close', () => {
                clearInterval(hb);
                const s = progressClients.get(key);
                if (s) s.delete(res);
            });
        } catch (e) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'failed to open sse' }));
        }
        return;
    }

    // 3Dモデル解析APIエンドポイント
    if (req.url === '/api/analyze-model' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', async () => {
            try {
                const { filePath } = JSON.parse(body);
                if (!filePath) { res.writeHead(400); res.end(JSON.stringify({ error: 'File path is required' })); return; }

                const baseDir = process.env.SCAN_PATH;
                if (!isPathSafe(filePath, baseDir)) { res.writeHead(403); res.end(JSON.stringify({ error: 'Invalid file path' })); return; }
                const fullPath = path.join(baseDir, filePath);
                if (!fs.existsSync(fullPath)) { res.writeHead(404); res.end(JSON.stringify({ error: 'File not found' })); return; }

                // Run model_indexer.py
                const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
                const modelScript = 'model_indexer.py';
                const pythonScript = path.join(__dirname, analysisDir, modelScript);
                const venvPython = VENV_PYTHON;

                const args = [pythonScript, '--model', fullPath];
                const child = spawn(venvPython, args, {
                    stdio: 'pipe',
                    env: {
                        ...process.env,
                        PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__')
                    }
                });

                let stdout = '', stderr = '';
                child.stdout.on('data', d => stdout += d.toString());
                child.stderr.on('data', d => stderr += d.toString());
                child.on('close', code => {
                    if (code !== 0) {
                        res.writeHead(500);
                        res.end(JSON.stringify({ error: 'Model analysis failed', details: stderr }));
                        return;
                    }
                    let analysisResult = null;
                    for (const line of stdout.split('\n')) {
                        if (line.startsWith('MODEL_ANALYSIS_RESULT: ')) {
                            try {
                                const jsonStr = line.substring('MODEL_ANALYSIS_RESULT: '.length);
                                analysisResult = JSON.parse(jsonStr);
                            } catch (e) {}
                            break;
                        }
                    }
                    res.writeHead(200);
                    res.end(JSON.stringify({ success: true, result: analysisResult }));
                });
                child.on('error', err => {
                    res.writeHead(500);
                    res.end(JSON.stringify({ error: 'Failed to start model analysis', details: err.message }));
                });
            } catch (err) {
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Invalid JSON request' }));
            }
        });
        return;
    }

    // 画像データ解析API（base64 PNGなどを受け取り一時ファイルとして解析）
    if (req.url === '/api/analyze-image-data' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', async () => {
            try {
                const { imageData, targetPath } = JSON.parse(body);
                if (!imageData || typeof imageData !== 'string') {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'imageData is required (data URL)' }));
                    return;
                }
                // data URL からデータを抽出
                const m = imageData.match(/^data:image\/(png|jpeg);base64,(.*)$/i);
                if (!m) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Unsupported image data URL' }));
                    return;
                }
                if (!targetPath || typeof targetPath !== 'string') {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'targetPath is required (relative to SCAN_PATH)' }));
                    return;
                }
                const baseDir = process.env.SCAN_PATH;
                if (!baseDir || !isPathSafe(targetPath, baseDir)) {
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Invalid targetPath' }));
                    return;
                }
                const ext = m[1].toLowerCase() === 'jpeg' ? 'jpg' : m[1].toLowerCase();
                const b64 = m[2];
                const buf = Buffer.from(b64, 'base64');
                // 保存先を SCAN_PATH 配下に
                const absTarget = path.join(baseDir, targetPath);
                const targetDir = path.dirname(absTarget);
                const shotDir = path.join(targetDir, '.snapshots');
                try { fs.mkdirSync(shotDir, { recursive: true }); } catch {}
                const baseName = path.basename(absTarget, path.extname(absTarget));
                const shotPath = path.join(shotDir, `${baseName}.preview.${ext}`);
                fs.writeFileSync(shotPath, buf);

                // 画像インデクサで登録（--rel_base指定）
                const analysisDir = process.env.ANALYSIS_SCRIPTS_DIR || 'analysis';
                const imageScript = process.env.IMAGE_INDEXER_SCRIPT || 'image_indexer.py';
                const pythonScript = path.join(__dirname, analysisDir, imageScript);
                const venvPython = VENV_PYTHON;
                // override_path に targetPath（SCAN_PATHからの相対）を渡し、登録は元のGLBパスで行う
                const args = [pythonScript, '--image', shotPath, '--rel_base', baseDir, '--override_path', targetPath];
                // GLB のスナップショット解析時は強制タグを付与
                if (/\.glb$/i.test(targetPath)) {
                    args.push('--extra_tag', 'glb');
                    args.push('--extra_tag', '3D');
                    // プロンプトに 3D スナップショットの文脈を追加し、背景や空などの誤解釈を抑制
                    args.push('--context', 'これは3Dモデルのスナップショット画像です。背景や空・鳥などの外部風景ではなく、画面内のオブジェクトそのものの形状・材質・色・構造・用途を具体的に説明してください。出力はJSONのみ (caption, tags) で返してください。');
                }
                const child = spawn(venvPython, args, { 
                    stdio: 'pipe', 
                    env: { 
                        ...process.env, 
                        PYTHONPYCACHEPREFIX: path.join(__dirname, process.env.PYTHON_CACHE_DIR || 'analysis/__pycache__')
                    } 
                });
                let stderr = '';
                let stdout = '';
                child.stderr.on('data', d => { const s = d.toString(); stderr += s; console.log('[analyze-image-data][stderr]', s); });
                child.stdout.on('data', d => { const s = d.toString(); stdout += s; console.log('[analyze-image-data][stdout]', s); });
                child.on('close', code => {
                    // スナップショットは解析後に常に削除（一時的なファイルのみ）
                    try { 
                        fs.unlinkSync(shotPath);
                        console.log('[analyze-image-data] Snapshot deleted:', shotPath);
                    } catch (e) {
                        console.log('[analyze-image-data] Could not delete snapshot:', shotPath, e.message);
                    }
                    if (code !== 0) {
                        res.writeHead(500);
                        res.end(JSON.stringify({ error: 'Image indexing failed', details: stderr }));
                        return;
                    }
                    // Try to extract RESULT_JSON from stdout like /api/analyze-image
                    let analysisResult = null;
                    if (stdout) {
                        const lines = stdout.split('\n');
                        for (const raw of lines) {
                            const idx = raw.indexOf('RESULT_JSON:');
                            if (idx !== -1) {
                                const jsonStr = raw.slice(idx + 'RESULT_JSON:'.length);
                                try { analysisResult = JSON.parse(jsonStr); } catch {}
                                break;
                            }
                        }
                    }
                    if (!analysisResult) {
                        console.warn('[analyze-image-data] RESULT_JSON not found in stdout');
                    } else {
                        console.log('[analyze-image-data] Parsed RESULT_JSON:', analysisResult);
                    }
                    res.writeHead(200);
                    res.end(JSON.stringify({ success: true, result: analysisResult }));
                });
            } catch (err) {
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Invalid JSON request' }));
            }
        });
        return;
    }
    
    // 検索APIエンドポイント
    if (req.url.startsWith('/api/search') && req.method === 'GET') {
        try {
            const urlParams = new URL(req.url, `http://${process.env.SERVER_HOST || '127.0.0.1'}:${process.env.PORT}`);
            const query = urlParams.searchParams.get('q') || '';
            const type = urlParams.searchParams.get('type') || 'text';
            const limit = parseInt(urlParams.searchParams.get('limit')) || 20;
            const folder = urlParams.searchParams.get('folder') || '';
            
            let results = await searchMedia(query, type, limit);

            // 選択中フォルダにスコープ（path はSCAN_PATHからの相対パス）
            if (folder) {
                results = results.filter(r => typeof r.path === 'string' && (r.path === folder || r.path.startsWith(folder + '/')));
            }
            
            res.writeHead(200);
            res.end(JSON.stringify({
                success: true,
                results: results,
                count: results.length
            }));
        } catch (err) {
            console.error('Search error:', err);
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Search failed' }));
        }
        return;
    }

    // Meilisearch ダイアグノスティクス: インデックス一覧
    if (req.url.startsWith('/api/meili/indexes') && req.method === 'GET') {
        try {
            const meiliUrl = process.env.MEILI_URL || 'http://127.0.0.1:17700';
            const meiliKey = process.env.MEILI_MASTER_KEY || '';
            const headers = meiliKey ? { 'Authorization': `Bearer ${meiliKey}` } : {};
            const resp = await fetch(`${meiliUrl}/indexes`, { headers });
            const text = await resp.text();
            res.writeHead(resp.ok ? 200 : 500, { 'Content-Type': 'application/json' });
            try { res.end(text); } catch { res.end(JSON.stringify({ ok: resp.ok, body: text })); }
        } catch (e) {
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Failed to reach Meilisearch', details: e.message }));
        }
        return;
    }

    // Meilisearch ダイアグノスティクス: インデックスstats
    if (req.url.startsWith('/api/meili/stats') && req.method === 'GET') {
        try {
            const base = new URL(req.url, `http://${process.env.SERVER_HOST || '127.0.0.1'}:${process.env.PORT}`);
            const index = base.searchParams.get('index') || (process.env.MEILI_INDEX || 'media');
            const meiliUrl = process.env.MEILI_URL || 'http://127.0.0.1:17700';
            const meiliKey = process.env.MEILI_MASTER_KEY || '';
            const headers = meiliKey ? { 'Authorization': `Bearer ${meiliKey}` } : {};
            const resp = await fetch(`${meiliUrl}/indexes/${encodeURIComponent(index)}/stats`, { headers });
            const text = await resp.text();
            res.writeHead(resp.ok ? 200 : 500, { 'Content-Type': 'application/json' });
            try { res.end(text); } catch { res.end(JSON.stringify({ ok: resp.ok, body: text })); }
        } catch (e) {
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Failed to reach Meilisearch', details: e.message }));
        }
        return;
    }

    // スナップショット保存API
    if (req.url === '/api/save-snapshot' && req.method === 'POST') {
        const multer = require('multer');
        const upload = multer({ 
            storage: multer.memoryStorage(),
            limits: { fileSize: 5 * 1024 * 1024 } // 5MB上限でDoS緩和
        });
        
        upload.single('snapshot')(req, res, async (err) => {
            if (err) {
                res.writeHead(500);
                res.end(JSON.stringify({ error: 'Upload failed' }));
                return;
            }
            
            try {
                const targetPath = req.body.targetPath;
                if (!targetPath || !req.file) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Missing parameters' }));
                    return;
                }
                
                const baseDir = process.env.SCAN_PATH;
                if (!baseDir) {
                    res.writeHead(500);
                    res.end(JSON.stringify({ error: 'SCAN_PATH not set' }));
                    return;
                }
                // パスバリデーション
                const normalized = String(targetPath).replace(/\\/g, '/');
                if (!normalized || normalized.startsWith('/') || normalized.includes('..')) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Invalid targetPath' }));
                    return;
                }
                if (!normalized.startsWith('.snapshots/')) {
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Snapshots must be under .snapshots/' }));
                    return;
                }
                const ext = path.extname(normalized).toLowerCase();
                if (ext !== '.png' && ext !== '.webp' && ext !== '.jpg' && ext !== '.jpeg') {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'Unsupported extension' }));
                    return;
                }
                if (!isPathSafe(normalized, baseDir)) {
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Path not allowed' }));
                    return;
                }
                // 保存先ディレクトリ作成
                const fullPath = path.join(baseDir, normalized);
                const saveDir = path.dirname(fullPath);
                try { fs.mkdirSync(saveDir, { recursive: true }); } catch {}
                // ファイル保存
                fs.writeFileSync(fullPath, req.file.buffer);
                
                res.writeHead(200);
                res.end(JSON.stringify({ success: true, path: normalized }));
                
            } catch (error) {
                console.error('Snapshot save error:', error);
                res.writeHead(500);
                res.end(JSON.stringify({ error: 'Save failed' }));
            }
        });
        return;
    }

    // メタデータ取得API（path指定でcaption/tagsを取得）
    if (req.url.startsWith('/api/metadata') && req.method === 'GET') {
        try {
            const urlParams = new URL(req.url, `http://${process.env.SERVER_HOST || '127.0.0.1'}:${process.env.PORT}`);
            const relPath = urlParams.searchParams.get('path') || '';
            if (!relPath) {
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'path is required' }));
                return;
            }

            const baseDir = process.env.SCAN_PATH;
            if (!baseDir) {
                res.writeHead(500);
                res.end(JSON.stringify({ error: 'SCAN_PATH environment variable is not set' }));
                return;
            }

            // パス検証と絶対パス化（Meiliには絶対パスで入っている想定）
            if (!isPathSafe(relPath, baseDir)) {
                console.warn(`[SECURITY] Path traversal in metadata: ${relPath}`);
                res.writeHead(403);
                res.end(JSON.stringify({ error: 'Invalid file path' }));
                return;
            }
            const absPath = path.join(baseDir, relPath);

            // まずはドキュメントIDを推定して直接取得
            // 優先: rel8（relPathのsha256先頭8hex→int）
            // 次点: rel16（旧画像方式）→ abs8（旧音声方式）
            const meiliUrl = process.env.MEILI_URL || 'http://127.0.0.1:17700';
            const meiliIndex = process.env.MEILI_INDEX || 'media';
            const meiliApiKey = process.env.MEILI_MASTER_KEY || 'masterKey';
            let mediaDoc = null;
            try {
                const idR8 = computeMediaIdRel8(relPath);
                const headers = { 'Authorization': `Bearer ${meiliApiKey}` };
                let docRes = await fetch(`${meiliUrl}/indexes/${meiliIndex}/documents/${idR8}` , { headers });
                if (docRes.ok) {
                    mediaDoc = await docRes.json();
                } else {
                    const idR16 = computeMediaIdRel16(relPath);
                    docRes = await fetch(`${meiliUrl}/indexes/${meiliIndex}/documents/${idR16}` , { headers });
                    if (docRes.ok) {
                        mediaDoc = await docRes.json();
                    } else {
                        // 互換: 旧音声のID方式（abs8）
                        const idA8 = computeMediaIdFromAbsPath(normalizePath(absPath));
                        docRes = await fetch(`${meiliUrl}/indexes/${meiliIndex}/documents/${idA8}`, { headers });
                        if (docRes.ok) mediaDoc = await docRes.json();
                    }
                }
            } catch (e) {
                // ignore and fallback
            }

            if (!mediaDoc) {
                // Fallback: 検索でベース名/相対/絶対などの順に探す
                const searchUrl = `${meiliUrl}/indexes/${meiliIndex}/search`;
                const baseName = path.basename(relPath);
                const response = await fetch(searchUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${meiliApiKey}`
                    },
                    body: JSON.stringify({
                        q: baseName,
                        limit: 50,
                        attributesToRetrieve: ['id', 'path', 'caption', 'tags', 'model', 'category', 'instruments']
                    })
                });
                if (response.ok) {
                    const json = await response.json();
                    const hits = Array.isArray(json.hits) ? json.hits : [];
                    const norm = (p) => p ? p.replace(/\\/g, '/'): '';
                    const targetAbs = norm(absPath);
                    const targetJoin = norm(path.join(baseDir, relPath));
                    const targetRel = norm(relPath);
                    const hit = hits.find(h => {
                        const hp = norm(h.path || '');
                        return hp === targetAbs
                            || hp === targetJoin
                            || hp === targetRel
                            || hp.endsWith('/' + targetRel)
                            || hp.endsWith(targetRel);
                    }) || null;
                    if (hit) mediaDoc = hit;
                }
            }

            if (!mediaDoc) {
                res.writeHead(200);
                res.end(JSON.stringify({ success: true, media: null }));
                return;
            }

            // 表示用に相対パスへ変換（区切りを正規化）
            let displayPath = mediaDoc.path;
            const normBase = normalizePath(baseDir || '');
            const normDoc = normalizePath(displayPath || '');
            if (normDoc && normBase && (normDoc === normBase || normDoc.startsWith(normBase + '/'))) {
                displayPath = normDoc.substring(normBase.length + (normDoc === normBase ? 0 : 1));
            } else if (displayPath) {
                displayPath = normDoc; // 正規化のみ
            }
            const media = {
                id: mediaDoc.id,
                path: displayPath,
                caption: mediaDoc.caption,
                tags: mediaDoc.tags,
                model: mediaDoc.model,
                category: mediaDoc.category,
                instruments: mediaDoc.instruments
            };
            res.writeHead(200);
            res.end(JSON.stringify({ success: true, media }));
        } catch (err) {
            console.error('Metadata error:', err);
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Failed to get metadata' }));
        }
        return;
    }
    
    // メディア詳細APIエンドポイント
    if (req.url.startsWith('/api/media/') && req.method === 'GET') {
        try {
            const mediaId = req.url.split('/api/media/')[1];
            if (!mediaId || isNaN(parseInt(mediaId))) {
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Invalid media ID' }));
                return;
            }
            
            const mediaData = await getMediaById(parseInt(mediaId));
            if (!mediaData) {
                res.writeHead(404);
                res.end(JSON.stringify({ error: 'Media not found' }));
                return;
            }
            
            res.writeHead(200);
            res.end(JSON.stringify({
                success: true,
                media: mediaData
            }));
        } catch (err) {
            console.error('Get media error:', err);
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Failed to get media details' }));
        }
        return;
    }
    
    if (req.url === '/api/scan' && req.method === 'GET') {
        // .envで指定された絶対パスをスキャン
        if (!process.env.SCAN_PATH) {
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'SCAN_PATH environment variable is not set' }));
            return;
        }
        const currentDir = process.env.SCAN_PATH;
        console.log('Scanning directory:', currentDir);
        const mediaFiles = scanDirectory(currentDir);
        
        res.writeHead(200);
        res.end(JSON.stringify({
            baseDir: currentDir,
            data: mediaFiles
        }));
    } else if (req.url === '/api/open-file' && req.method === 'POST') {
        // ファイルを安全に開く
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', async () => {
            try {
                const { path: filePath } = JSON.parse(body);
                if (!filePath) {
                    res.writeHead(400);
                    res.end(JSON.stringify({ error: 'File path is required' }));
                    return;
                }
                
                if (!process.env.SCAN_PATH) {
                    res.writeHead(500);
                    res.end(JSON.stringify({ error: 'SCAN_PATH environment variable is not set' }));
                    return;
                }
                
                const baseDir = process.env.SCAN_PATH;
                
                // パストラバーサル攻撃をチェック
                if (!isPathSafe(filePath, baseDir)) {
                    console.warn(`[SECURITY] Path traversal in open-file: ${filePath}`);
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Invalid file path' }));
                    return;
                }
                
                const fullPath = path.join(baseDir, filePath);
                
                // spawn使用でシェルインジェクション回避
                let command, args;
                if (process.platform === 'darwin') {
                    command = 'open';
                    args = [fullPath];
                } else if (process.platform === 'win32') {
                    command = 'cmd';
                    args = ['/c', 'start', '', fullPath];
                } else {
                    command = 'xdg-open';
                    args = [fullPath];
                }
                
                const child = spawn(command, args, { stdio: 'ignore', detached: true });
                child.unref();
                
                child.on('error', (error) => {
                    console.error('Error opening file:', error);
                    res.writeHead(500);
                    res.end(JSON.stringify({ error: 'Failed to open file' }));
                });
                
                res.writeHead(200);
                res.end(JSON.stringify({ success: true }));
                
            } catch (err) {
                console.error('Error parsing request:', err);
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Invalid request' }));
            }
        });
    } else if (req.url === '/api/open-folder' && req.method === 'POST') {
        // フォルダを開く（パス検証＋spawnでシェルインジェクション回避）
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', () => {
            try {
                const { path: folderPath } = JSON.parse(body);
                const baseDir = process.env.SCAN_PATH;
                if (!baseDir) {
                    res.writeHead(500);
                    res.end(JSON.stringify({ error: 'SCAN_PATH environment variable is not set' }));
                    return;
                }
                if (!folderPath || !isPathSafe(folderPath, baseDir)) {
                    res.writeHead(403);
                    res.end(JSON.stringify({ error: 'Invalid folder path' }));
                    return;
                }
                const fullPath = path.join(baseDir, folderPath);

                let command, args;
                if (process.platform === 'darwin') {
                    command = 'open';
                    args = [fullPath];
                } else if (process.platform === 'win32') {
                    command = 'cmd';
                    args = ['/c', 'start', '', fullPath];
                } else {
                    command = 'xdg-open';
                    args = [fullPath];
                }
                const child = spawn(command, args, { stdio: 'ignore', detached: true });
                child.unref();
                child.on('error', (error) => {
                    console.error('Error opening folder:', error);
                    res.writeHead(500);
                    res.end(JSON.stringify({ error: 'Failed to open folder' }));
                });
                res.writeHead(200);
                res.end(JSON.stringify({ success: true }));
            } catch (err) {
                console.error('Error parsing request:', err);
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Invalid request' }));
            }
        });
    } else {
        // Qdrantコレクションをリセット（削除）
        if (req.url === '/api/qdrant/reset' && req.method === 'POST') {
            try {
                const qdrantUrl = process.env.QDRANT_URL || 'http://127.0.0.1:26333';
                const collection = process.env.QDRANT_COLLECTION || 'media_vectors';
                const delUrl = `${qdrantUrl}/collections/${collection}`;
                const resp = await fetch(delUrl, { method: 'DELETE' });
                const text = await resp.text();
                res.writeHead(resp.ok ? 200 : 500);
                res.end(JSON.stringify({ success: resp.ok, response: text }));
            } catch (e) {
                res.writeHead(500);
                res.end(JSON.stringify({ success: false, error: e.message }));
            }
            return;
        }

        res.writeHead(404);
        res.end(JSON.stringify({ error: 'Not found' }));
    }
});

if (!process.env.PORT) {
    console.error('ERROR: PORT environment variable is not set');
    process.exit(1);
}
const PORT = process.env.PORT;
server.listen(PORT, () => {
    console.log(`Media scanner server running at http://${process.env.SERVER_HOST || '127.0.0.1'}:${PORT}`);
    console.log(`API endpoint: http://${process.env.SERVER_HOST || '127.0.0.1'}:${PORT}/api/scan`);
});

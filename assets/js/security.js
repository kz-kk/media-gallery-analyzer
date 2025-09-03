// セキュリティ関数群（ESM）
import { analyzeImage, analyzeVideo, analyzeAudio, analyzeModelAppearance } from './analysis.js';
import { apiBaseUrl } from './api.js';
import { getMetaForPath, setMetaForPath } from './state.js';
import { DOMPurify } from './purify.esm.js';

// HTML文字列のエスケープ (DOMPurify使用)
export function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    return DOMPurify.sanitize(text, { ALLOWED_TAGS: [] });
}

// HTML文字列をサニタイズ（タグを許可する場合）
export function sanitizeHtml(html, options = {}) {
    return DOMPurify.sanitize(html, {
        ALLOWED_TAGS: options.allowedTags || ['b', 'i', 'em', 'strong', 'span', 'div'],
        ALLOWED_ATTR: options.allowedAttributes || ['class', 'style'],
        ...options
    });
}

// 安全な要素作成関数
export function createSafeElement(tagName, options = {}) {
    const element = document.createElement(tagName);
    if (options.className) element.className = options.className;
    if (options.textContent) element.textContent = options.textContent;
    if (options.src) element.src = options.src;
    if (options.alt) element.alt = options.alt;
    if (options.style) element.style.cssText = options.style;
    return element;
}

// XSS安全なメディアアイテム作成関数
export function createSafeMediaItem(file, options = {}) {
    const showMeta = !!options.showMeta; // 検索結果などでのみメタ表示
    const item = document.createElement('div');
    item.className = `media-item ${file.type}`;
    item.dataset.path = file.path;
    item.dataset.name = file.name;
    item.dataset.type = file.type;
    
    if (file.type === 'image') {
        // 画像サムネイル
        const img = createSafeElement('img', {
            className: 'media-thumbnail',
            src: file.path,
            alt: file.name
        });
        img.loading = 'lazy';
        
        // 解析ボタン
        const analyzeBtn = createSafeElement('button', {
            className: 'analyze-button',
            textContent: '解析'
        });
        analyzeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            analyzeImage(file.path, e.currentTarget);
        });
        
        item.appendChild(img);
        item.appendChild(analyzeBtn);
        
    } else if (file.type === 'video') {
        // 動画コンテナ
        const videoContainer = createSafeElement('div', {
            style: 'position: relative;'
        });
        
        const video = createSafeElement('video', {
            className: 'media-thumbnail',
            src: file.path
        });
        video.muted = true;
        video.preload = 'metadata';
        
        const overlay = createSafeElement('div', {
            className: 'video-overlay'
        });
        const playButton = createSafeElement('div', {
            className: 'play-button'
        });
        overlay.appendChild(playButton);
        
        const analyzeBtn = createSafeElement('button', {
            className: 'analyze-button',
            textContent: '解析'
        });
        analyzeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            analyzeVideo(file.path, e.currentTarget);
        });
        
        videoContainer.appendChild(video);
        videoContainer.appendChild(overlay);
        item.appendChild(videoContainer);
        item.appendChild(analyzeBtn);
    } else if (file.type === 'audio') {
        // 音声ファイルサムネイル
        const musicIcons = ['🎵', '🎶', '🎼', '🎤', '🎧', '🎸', '🎹', '🎺', '🎷', '🥁'];
        const colors = [
            'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
            'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
            'linear-gradient(135deg, #30cfd0 0%, #330867 100%)'
        ];
        const randomIcon = musicIcons[Math.floor(Math.random() * musicIcons.length)];
        const randomColor = colors[Math.floor(Math.random() * colors.length)];
        
        const audioThumbnail = createSafeElement('div', {
            className: 'media-thumbnail audio',
            style: `background: ${randomColor};`
        });
        
        const iconWrapper = createSafeElement('div', {
            className: 'audio-icon-wrapper'
        });
        const mainIcon = createSafeElement('span', {
            className: 'main-icon',
            textContent: randomIcon
        });
        iconWrapper.appendChild(mainIcon);
        
        const waveBars = createSafeElement('div', {
            className: 'audio-wave-bars'
        });
        for (let i = 0; i < 7; i++) {
            const bar = createSafeElement('span');
            waveBars.appendChild(bar);
        }
        
        audioThumbnail.appendChild(iconWrapper);
        audioThumbnail.appendChild(waveBars);
        item.appendChild(audioThumbnail);

        if (showMeta) {
            // メタ情報エリア（名前・キャプション・タグ）
            const infoWrap = createSafeElement('div', { className: 'media-info' });
            const nameEl = createSafeElement('div', { className: 'media-name', textContent: file.name });
            const captionEl = createSafeElement('div', { className: 'media-details', style: 'font-size:.75rem;color:#999;' });
            const tagsEl = createSafeElement('div', { className: 'media-details', style: 'font-size:.7rem;color:#666;' });
            captionEl.textContent = (typeof file.caption === 'string' && file.caption.trim())
                ? file.caption.substring(0, 50) + (file.caption.length > 50 ? '...' : '')
                : 'キャプションなし';
            const initialTags = Array.isArray(file.tags) ? file.tags.join(', ') : (typeof file.tags === 'string' ? file.tags : '');
            tagsEl.textContent = initialTags || 'タグなし';
            infoWrap.appendChild(nameEl);
            infoWrap.appendChild(captionEl);
            infoWrap.appendChild(tagsEl);
            item.appendChild(infoWrap);

            // 既存キャッシュまたはAPIからメタデータを取得して反映
            (async () => {
                try {
                    const cached = getMetaForPath(file.path);
                    if (cached && (cached.caption || cached.tags)) {
                        if (cached.caption) {
                            const c = String(cached.caption);
                            captionEl.textContent = c.substring(0, 50) + (c.length > 50 ? '...' : '');
                        }
                        if (cached.tags) {
                            const t = Array.isArray(cached.tags) ? cached.tags.join(', ') : String(cached.tags);
                            tagsEl.textContent = t || 'タグなし';
                        }
                        return;
                    }
                    const resp = await fetch(`${apiBaseUrl}/api/metadata?path=${encodeURIComponent(file.path)}`);
                    const data = await resp.json().catch(() => ({}));
                    if (data && data.success && data.media) {
                        setMetaForPath(file.path, { caption: data.media.caption, tags: data.media.tags });
                        const c = data.media.caption ? String(data.media.caption) : '';
                        captionEl.textContent = c ? (c.substring(0, 50) + (c.length > 50 ? '...' : '')) : 'キャプションなし';
                        const t = Array.isArray(data.media.tags) ? data.media.tags.join(', ') : (data.media.tags || '');
                        tagsEl.textContent = t || 'タグなし';
                    }
                } catch (e) {
                    // 失敗時はそのまま（プレースホルダ表示）
                }
            })();
        }

        // 音声解析ボタン
        const analyzeBtn = createSafeElement('button', {
            className: 'analyze-button',
            textContent: '解析'
        });
        analyzeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            analyzeAudio(file.path, e.currentTarget);
        });
        item.appendChild(analyzeBtn);
    } else if (file.type === 'model') {
        // 3Dモデルサムネイル
        const thumb = createSafeElement('div', {
            className: 'media-thumbnail model',
            style: 'display:flex;align-items:center;justify-content:center;font-size:42px;color:#888;overflow:hidden;background:#111;'
        });
        // スナップショット（.snapshots/<path_replaced>.(webp|png)）があればそれを表示。なければ3Dアイコン
        const label = createSafeElement('div', { className: 'model-label', textContent: '🧊 3D' });
        const snapshotBase = `.snapshots/${file.path.replace(/[\\/\\]/g, '_')}`;
        const snapshotWebp = `${snapshotBase}.webp`;
        const snapshotPng = `${snapshotBase}.png`;
        let activeSnapshotPath = null;

        // まずは常に3Dアイコンを表示
        thumb.appendChild(label);
        item.appendChild(thumb);

        // 解析ボタンを追加（必要なら再生成）
        function addAnalyzeButton() {
            if (!thumb.querySelector('.analyze-button')) {
                const btn = createSafeElement('button', { className: 'analyze-button', textContent: '解析' });
                btn.style.position = 'absolute';
                btn.style.right = '10px';
                btn.style.bottom = '10px';
                btn.style.zIndex = '5';
                btn.addEventListener('click', async (ev) => {
                    ev.stopPropagation();
                    // 解析前にスナップショットを生成（無い場合）
                    const snapshotBase = `.snapshots/${file.path.replace(/[\\/\\]/g, '_')}`;
                    const snapshotWebp = `${snapshotBase}.webp`;
                    const snapshotPng = `${snapshotBase}.png`;
                    let needSnapshot = true;
                    try {
                        let resp = await fetch(snapshotWebp, { method: 'HEAD' });
                        if (resp.status === 200) needSnapshot = false;
                        else {
                            resp = await fetch(snapshotPng, { method: 'HEAD' });
                            needSnapshot = resp.status !== 200;
                        }
                    } catch (_) { /* keep needSnapshot = true */ }

                if (needSnapshot) {
                    const mediaItem = btn.closest('.media-item');
                    let overlay = null;
                    if (mediaItem) {
                        overlay = document.createElement('div');
                        overlay.className = 'analysis-overlay';
                        overlay.innerHTML = '<div class="analysis-spinner"></div><div>スナップショット生成中...</div>';
                        mediaItem.style.position = 'relative';
                        mediaItem.appendChild(overlay);
                    }
                    // 一時viewerを作成してスナップショット
                        const mv = document.createElement('model-viewer');
                        mv.style.width = '1024px';
                        mv.style.height = '1024px';
                        mv.style.position = 'fixed';
                        mv.style.left = '0px';
                        mv.style.top = '0px';
                        mv.style.opacity = '0';
                        mv.style.pointerEvents = 'none';
                        mv.style.zIndex = '-1';
                        mv.setAttribute('reveal', 'auto');
                        mv.setAttribute('disable-zoom', '');
                        mv.setAttribute('disable-pan', '');
                        mv.setAttribute('environment-image', 'neutral');
                        mv.setAttribute('auto-rotate', '');
                        mv.src = file.path;
                        document.body.appendChild(mv);
                        await new Promise((resolve) => {
                            let to = setTimeout(resolve, 8000);
                            mv.addEventListener('load', () => { clearTimeout(to); resolve(); }, { once: true });
                        });
                        await new Promise(r => setTimeout(r, 600));
                        try {
                            const bigBlob = await mv.toBlob({ mimeType: 'image/png', qualityArgument: 0.85 });
                            if (bigBlob) {
                                const toWebpSameSize = (blob, quality = 0.5) => new Promise(async (resolve) => {
                                    const url = URL.createObjectURL(blob);
                                    const img = new Image();
                                    img.onload = () => {
                                        const tw = img.width;
        							const th = img.height;
                                        const canvas = document.createElement('canvas');
                                        canvas.width = tw; canvas.height = th;
                                        const ctx = canvas.getContext('2d');
                                        ctx.drawImage(img, 0, 0, tw, th);
                                        canvas.toBlob((out) => { URL.revokeObjectURL(url); resolve(out); }, 'image/webp', quality);
                                    };
                                    img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
                                    img.src = url;
                                });
                                const smallBlob = await toWebpSameSize(bigBlob, 0.5);
                                const fd = new FormData();
                                fd.append('snapshot', smallBlob || bigBlob, smallBlob ? 'snapshot.webp' : 'snapshot.png');
                                fd.append('targetPath', smallBlob ? snapshotWebp : snapshotPng);
                                const resp = await fetch('/api/save-snapshot', { method: 'POST', body: fd });
                                if (resp.ok) {
                                    // 一覧の該当サムネを更新
                                    await refreshSnapshotThumb();
                                    // グローバル通知（他ビューにも反映）
                                    window.dispatchEvent(new CustomEvent('snapshot-saved', { detail: { path: file.path } }));
                                }
                            }
                    } catch (_) { /* ignore */ }
                    finally { 
                        if (mv && mv.parentElement) mv.parentElement.removeChild(mv);
                        // overlay は解析へ引き継ぐため、ここでは消さない
                    }
                }
                analyzeModelAppearance(file.path, ev.currentTarget);
                });
                thumb.style.position = 'relative';
                thumb.appendChild(btn);
            }
        }

        // スナップショットを確認してサムネに反映（webp優先）
        async function refreshSnapshotThumb() {
            try {
                let resp = await fetch(snapshotWebp, { method: 'HEAD' });
                if (resp.status === 200) {
                    activeSnapshotPath = snapshotWebp;
                } else {
                    resp = await fetch(snapshotPng, { method: 'HEAD' });
                    if (resp.status === 200) activeSnapshotPath = snapshotPng;
                }
                if (activeSnapshotPath) {
                    const img = new Image();
                    img.className = 'media-thumbnail';
                    img.style.objectFit = 'cover';
                    img.style.width = '100%';
                    img.style.height = '100%';
                    img.onload = () => {
                        thumb.innerHTML = '';
                        thumb.appendChild(img);
                        thumb.style.position = 'relative';
                        addAnalyzeButton();
                    };
                    img.src = activeSnapshotPath + `?t=${Date.now()}`;
                }
            } catch (e) { /* ignore */ }
        }
        // 初回チェック
        refreshSnapshotThumb();
        // グローバル通知で即時反映
        const onSnapshotSaved = (ev) => {
            if (ev && ev.detail && ev.detail.path === file.path) {
                refreshSnapshotThumb();
            }
        };
        window.addEventListener('snapshot-saved', onSnapshotSaved);

        // 解析ボタンを常時表示（右下）
        addAnalyzeButton();
    }
    
    // メディア情報
    const mediaInfo = createSafeElement('div', {
        className: 'media-info'
    });
    const mediaName = createSafeElement('div', {
        className: 'media-name',
        textContent: file.name
    });
    mediaInfo.appendChild(mediaName);
    item.appendChild(mediaInfo);
    // クリックイベントは呼び出し元（gallery/ui-handlers）で付与
    
    return item;
}

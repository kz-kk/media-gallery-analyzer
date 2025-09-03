// UI イベントハンドラー（ESM）
import { openLightbox, closeLightbox } from './lightbox.js';
import { showContextMenu, hideContextMenu } from './context-menu.js';
import { analyzeImage, analyzeVideo, analyzeAudio, analyzeModelAppearance } from './analysis.js';
import { searchMedia } from './search.js';
import { allMediaFiles, setSelectedFile, getCurrentFilter } from './state.js';
import { displayMedia, filterByType } from './gallery.js';
import { scanDirectory } from './api.js';
import { sanitizeHtml, escapeHtml } from './security.js';

// 追加のメディアを読み込む
export function loadMoreMedia(startIndex) {
    const grid = document.getElementById('mediaGrid');
    // 既存のロードモア要素をすべて削除
    grid.querySelectorAll('.load-more, .load-more-btn').forEach(el => el.remove());
    
    const cf = getCurrentFilter();
    const files = cf === 'all' ? allMediaFiles : allMediaFiles.filter(f => f.type === cf);
    const fragment = document.createDocumentFragment();
    const endIndex = Math.min(startIndex + 50, files.length);
    
    for (let i = startIndex; i < endIndex; i++) {
        const file = files[i];
        const item = document.createElement('div');
        item.className = 'media-item';
        item.dataset.path = file.path;
        item.dataset.type = file.type;
        
        if (file.type === 'image') {
            item.className = 'media-item image';
            item.innerHTML = sanitizeHtml(`
                <img class="media-thumbnail" src="${escapeHtml(file.path)}" alt="${escapeHtml(file.name)}" loading="lazy">
                <button class="analyze-button">解析</button>
                <div class="media-info">
                    <div class="media-name">${escapeHtml(file.name)}</div>
                    <div class="media-details">Image</div>
                </div>
            `, { 
                ALLOWED_TAGS: ['img', 'button', 'div'], 
                ALLOWED_ATTR: ['class', 'src', 'alt', 'loading'] 
            });
        } else if (file.type === 'video') {
            item.innerHTML = sanitizeHtml(`
                <div style="position: relative;">
                    <video class="media-thumbnail" src="${escapeHtml(file.path)}" muted preload="metadata"></video>
                    <button class="analyze-button">解析</button>
                </div>
                <div class="media-info">
                    <div class="media-name">${escapeHtml(file.name)}</div>
                    <div class="media-details">Video</div>
                </div>
            `, { 
                ALLOWED_TAGS: ['div', 'video', 'button'], 
                ALLOWED_ATTR: ['class', 'style', 'src', 'muted', 'preload'] 
            });
            const video = item.querySelector('video');
            if (video) {
                video.addEventListener('loadedmetadata', () => {
                    video.currentTime = 0.1;
                });
            }
        } else if (file.type === 'audio') {
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
            item.innerHTML = sanitizeHtml(`
                <div class="media-thumbnail audio" style="background: ${escapeHtml(randomColor)};">
                    <div class="audio-icon-wrapper">
                        <span class="main-icon">${escapeHtml(randomIcon)}</span>
                    </div>
                    <div class="audio-wave-bars">
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                <button class="analyze-button">解析</button>
                <div class="media-info">
                    <div class="media-name">${escapeHtml(file.name)}</div>
                    <div class="media-details">Audio</div>
                </div>
            `, { 
                ALLOWED_TAGS: ['div', 'span', 'button'], 
                ALLOWED_ATTR: ['class', 'style'] 
            });
        } else if (file.type === 'model') {
            item.className = 'media-item model';
            item.innerHTML = sanitizeHtml(`
                <div class="media-thumbnail model" style="display:flex;align-items:center;justify-content:center;font-size:42px;color:#888;">
                    <div class="model-label">🧊 3D</div>
                </div>
                <div class="media-info">
                    <div class="media-name">${escapeHtml(file.name)}</div>
                    <div class="media-details">3D Model</div>
                </div>
            `, { 
                ALLOWED_TAGS: ['div'], 
                ALLOWED_ATTR: ['class', 'style'] 
            });

            const thumb = item.querySelector('.media-thumbnail.model');
            // スナップショット（webp優先）を見つけたら即座に差し替える
            const snapshotBase = `.snapshots/${file.path.replace(/[\\/\\]/g, '_')}`;
            const snapshotWebp = `${snapshotBase}.webp`;
            const snapshotPng = `${snapshotBase}.png`;
            async function refreshSnapshotThumb() {
                try {
                    let resp = await fetch(snapshotWebp, { method: 'HEAD' });
                    let active = null;
                    if (resp.status === 200) active = snapshotWebp; else {
                        resp = await fetch(snapshotPng, { method: 'HEAD' });
                        if (resp.status === 200) active = snapshotPng;
                    }
                    if (active) {
                        const img = new Image();
                        img.className = 'media-thumbnail';
                        img.style.objectFit = 'cover';
                        img.style.width = '100%';
                        img.style.height = '100%';
                        img.onload = () => { 
                            thumb.innerHTML = ''; 
                            thumb.appendChild(img); 
                            thumb.style.position = 'relative';
                            // サムネ更新後に解析ボタンを再度重ねる
                            if (!thumb.querySelector('.analyze-button')) {
                                thumb.appendChild(btn);
                            }
                        };
                        img.src = active + `?t=${Date.now()}`;
                    }
                } catch (_) { /* ignore */ }
            }
            refreshSnapshotThumb();
            window.addEventListener('snapshot-saved', (ev) => {
                if (ev && ev.detail && ev.detail.path === file.path) refreshSnapshotThumb();
            });

            // 解析ボタンを常時表示（生成も担当）
            const btn = document.createElement('button');
            btn.className = 'analyze-button';
            btn.textContent = '解析';
            btn.style.position = 'absolute';
            btn.style.right = '10px';
            btn.style.bottom = '10px';
            btn.style.zIndex = '5';
            btn.addEventListener('click', async (ev) => {
                ev.stopPropagation();
                // スナップショットが無ければ生成
                let need = true;
                try {
                    let r = await fetch(snapshotWebp, { method: 'HEAD' });
                    if (r.status === 200) need = false; else {
                        r = await fetch(snapshotPng, { method: 'HEAD' });
                        need = r.status !== 200;
                    }
                } catch (_) {}
                if (need) {
                    const mediaItem = btn.closest('.media-item');
                    let overlay = null;
                    if (mediaItem) {
                        overlay = document.createElement('div');
                        overlay.className = 'analysis-overlay';
                        overlay.innerHTML = '<div class="analysis-spinner"></div><div>スナップショット生成中...</div>';
                        mediaItem.style.position = 'relative';
                        mediaItem.appendChild(overlay);
                    }
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
                                refreshSnapshotThumb();
                                window.dispatchEvent(new CustomEvent('snapshot-saved', { detail: { path: file.path } }));
                            }
                        }
                    } catch (_) {}
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
        
        // イベントリスナー追加
        item.addEventListener('click', () => openLightbox(file));
        item.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            setSelectedFile(file);
            showContextMenu(e.pageX, e.pageY);
        });

        const analyzeBtn = item.querySelector('.analyze-button');
        if (analyzeBtn && file.type !== 'model') {
            analyzeBtn.addEventListener('click', (ev) => {
                ev.stopPropagation();
                if (file.type === 'video') analyzeVideo(file.path, ev.currentTarget);
                else if (file.type === 'audio') analyzeAudio(file.path, ev.currentTarget);
                else analyzeImage(file.path, ev.currentTarget);
            });
        }
        
        if (file.type === 'video') {
            const video = item.querySelector('video');
            if (video) {
                item.addEventListener('mouseenter', () => {
                    video.play().catch(err => console.log('Play failed:', err));
                });
                item.addEventListener('mouseleave', () => {
                    video.pause();
                    video.currentTime = 0.1;
                });
            }
        }
        
        fragment.appendChild(item);
    }
    
    grid.appendChild(fragment);
    
    // まだ残りがある場合
    if (endIndex < files.length) {
        const container = document.createElement('div');
        container.className = 'load-more';
        const btn = document.createElement('button');
        btn.className = 'btn-load-more';
        const remaining = files.length - endIndex;
        btn.innerHTML = `<span class="label">さらに読み込む (${remaining} 件)</span>`;
        btn.addEventListener('click', () => { btn.classList.add('loading'); loadMoreMedia(endIndex); });
        container.appendChild(btn);
        // const hint = document.createElement('div');
        // hint.className = 'load-more-hint';
        // hint.textContent = 'スクロールしても読み込めます';
        // container.appendChild(hint);
        grid.appendChild(container);
    }
}

// イベントリスナーの初期化
export function initializeEventListeners() {
    // 検索機能（search.jsのsearchMediaを使用）
    document.getElementById('searchInput').addEventListener('input', function(e) {
        if (window.__searchTimeout) clearTimeout(window.__searchTimeout);
        window.__searchTimeout = setTimeout(() => {
            const searchTypeEl = document.querySelector('input[name="searchType"]:checked');
            const searchType = searchTypeEl ? searchTypeEl.value : 'filename';
            searchMedia(e.target.value, searchType);
        }, 300);
    });

    // ビュー切り替え
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            if (this.dataset.view) {
                document.querySelectorAll('.view-btn[data-view]').forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                const grid = document.getElementById('mediaGrid');
                const items = grid.querySelectorAll('.media-item');

                if (this.dataset.view === 'list') {
                    grid.classList.add('list-view');
                    items.forEach(item => item.classList.add('list-view'));
                } else {
                    grid.classList.remove('list-view');
                    items.forEach(item => item.classList.remove('list-view'));
                }
            }
        });
    });

    // 検索タイプ変更時にテキストフィールドをクリア
    document.querySelectorAll('input[name="searchType"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const searchInput = document.getElementById('searchInput');
            searchInput.value = '';
            displayMedia(allMediaFiles);
            document.getElementById('statsInfo').textContent = 'Showing all files';
        });
    });

    // タグフィルター
    document.querySelectorAll('.tag').forEach(tag => {
        tag.addEventListener('click', function() {
            document.querySelectorAll('.tag').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            filterByType(this.dataset.type);
        });
    });

    // ESCキーでライトボックスを閉じる
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeLightbox();
        }
    });

    // ライトボックス背景クリックで閉じる
    document.getElementById('lightbox').addEventListener('click', function(e) {
        if (e.target === this) {
            closeLightbox();
        }
    });

    // クリックでコンテキストメニューを閉じる
    document.addEventListener('click', () => {
        hideContextMenu();
    });

    // リフレッシュ/解析ボタン
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) refreshBtn.addEventListener('click', () => scanDirectory());

    const analyzeAllBtn = document.getElementById('analyzeAllBtn');
    if (analyzeAllBtn) analyzeAllBtn.addEventListener('click', () => import('./analysis.js').then(m => m.analyzeAllImages()));

    const analyzeAllVideosBtn = document.getElementById('analyzeAllVideosBtn');
    if (analyzeAllVideosBtn) analyzeAllVideosBtn.addEventListener('click', () => import('./analysis.js').then(m => m.analyzeAllVideos()));

    // ライトボックス閉じる
    const lbClose = document.getElementById('lightboxCloseBtn');
    if (lbClose) lbClose.addEventListener('click', () => closeLightbox());

    // コンテキストメニューのアクション
    const ctxCopy = document.getElementById('ctxCopy');
    if (ctxCopy) ctxCopy.addEventListener('click', () => import('./context-menu.js').then(m => m.copyFilePath()));
    const ctxOpenFile = document.getElementById('ctxOpenFile');
    if (ctxOpenFile) ctxOpenFile.addEventListener('click', () => import('./api.js').then(m => m.openFile()).finally(() => hideContextMenu()));
    const ctxOpenFolder = document.getElementById('ctxOpenFolder');
    if (ctxOpenFolder) ctxOpenFolder.addEventListener('click', () => import('./api.js').then(m => m.openFolder()).finally(() => hideContextMenu()));

    // 解析モーダル閉じる
    const modalClose = document.getElementById('analysisModalClose');
    if (modalClose) modalClose.addEventListener('click', () => import('./analysis.js').then(m => m.closeAnalysisModal()));
}

// DOMContentLoaded時の初期化
export function initializeApp() {
    initializeEventListeners();
    scanDirectory();
}

// 初期化フックはindex.htmlの最小スニペットで呼び出し

// 検索関連（ESM）
import { apiBaseUrl } from './api.js';
import { displayMedia } from './gallery.js';
import { createSafeElement, sanitizeHtml, escapeHtml, createSafeMediaItem } from './security.js';
import { openLightbox } from './lightbox.js';
import { showContextMenu } from './context-menu.js';
import { setSelectedFile, allMediaFiles, getCurrentFolderPath } from './state.js';

export async function searchMedia(query, type = 'filename') {
    const grid = document.getElementById('mediaGrid');
    const statsInfo = document.getElementById('statsInfo');
    const folderScope = getCurrentFolderPath() || '';

    // 空クエリ時の挙動をタイプごとに分岐
    if (!query.trim()) {
        if (type === 'fulltext') {
            // デフォルトの全文検索を実行（サーバーに委譲）
            showSearchLoading(grid, statsInfo, '全文検索中...', type);
            try {
                const folderParam = folderScope ? `&folder=${encodeURIComponent(folderScope)}` : '';
                const response = await fetch(`${apiBaseUrl}/api/search?q=&type=fulltext&limit=50${folderParam}`);
                const result = await response.json();
                if (result.success) {
                    displaySearchResults(result.results || [], 'fulltext');
                } else {
                    // 失敗時は通常表示にフォールバック
                    displayMedia(allMediaFiles);
                    statsInfo.textContent = 'Showing all files';
                }
            } catch (error) {
                console.error('Full-text default search error:', error);
                // エラー時は通常表示にフォールバック
                displayMedia(allMediaFiles);
                statsInfo.textContent = 'Showing all files';
            }
            return;
        }
        // それ以外は検索をクリアして通常表示に戻す
        displayMedia(allMediaFiles);
        statsInfo.textContent = `Showing all files`;
        return;
    }

    if (type === 'filename') {
        // ファイル名検索（クライアント側）
        const filtered = allMediaFiles
            .filter(file => !folderScope || file.path.startsWith(folderScope))
            .filter(file => file.name.toLowerCase().includes(query.toLowerCase()));
        displaySearchResults(filtered, 'filename');
    } else if (type === 'database') {
        // データベース検索（サーバー側）
        showSearchLoading(grid, statsInfo, '検索中...', type);
        try {
            const folderParam = folderScope ? `&folder=${encodeURIComponent(folderScope)}` : '';
            const response = await fetch(`${apiBaseUrl}/api/search?q=${encodeURIComponent(query)}&type=text&limit=50${folderParam}`);
            const result = await response.json();
            if (result.success) {
                displaySearchResults(result.results || [], 'database');
            }
        } catch (error) {
            console.error('Search error:', error);
            statsInfo.textContent = `検索エラー: ${error.message}`;
            grid.innerHTML = '';
            const errorDiv = createSafeElement('div', {
                textContent: `検索エラー: ${error.message}`,
                style: 'padding: 20px; color: #ff6b6b;'
            });
            grid.appendChild(errorDiv);
        }
    } else if (type === 'fulltext') {
        // 全文検索（Meilisearch）
        showSearchLoading(grid, statsInfo, '全文検索中...', type);
        try {
            const folderParam = folderScope ? `&folder=${encodeURIComponent(folderScope)}` : '';
            const response = await fetch(`${apiBaseUrl}/api/search?q=${encodeURIComponent(query)}&type=fulltext&limit=50${folderParam}`);
            const result = await response.json();
            if (result.success) {
                displaySearchResults(result.results || [], 'fulltext');
            }
        } catch (error) {
            console.error('Full-text search error:', error);
            statsInfo.textContent = `全文検索エラー: ${error.message}`;
            grid.innerHTML = '';
            const errorDiv = createSafeElement('div', {
                textContent: `全文検索エラー: ${error.message}`,
                style: 'padding: 20px; color: #ff6b6b;'
            });
            grid.appendChild(errorDiv);
        }
    } else if (type === 'semantic') {
        // セマンティック検索（Qdrant）
        showSearchLoading(grid, statsInfo, 'セマンティック検索中...', type);
        try {
            const folderParam = folderScope ? `&folder=${encodeURIComponent(folderScope)}` : '';
            const response = await fetch(`${apiBaseUrl}/api/search?q=${encodeURIComponent(query)}&type=semantic&limit=50${folderParam}`);
            const result = await response.json();
            if (result.success) {
                displaySearchResults(result.results || [], 'semantic');
            }
        } catch (error) {
            console.error('Semantic search error:', error);
            statsInfo.textContent = `セマンティック検索エラー: ${error.message}`;
            grid.innerHTML = '';
            const errorDiv = createSafeElement('div', {
                textContent: `セマンティック検索エラー: ${error.message}`,
                style: 'padding: 20px; color: #ff6b6b;'
            });
            grid.appendChild(errorDiv);
        }
    }
}

export function showSearchLoading(grid, statsInfo, message, searchType) {
    const loadingColor = searchType === 'semantic' ? '#9333ea' : '#4a9eff';
    statsInfo.textContent = message;
    grid.innerHTML = `
        <div style="position:absolute;top:0;left:0;right:0;bottom:0;display:flex;flex-direction:column;justify-content:center;align-items:center;background:rgba(26,26,26,0.8);backdrop-filter:blur(5px);color:${loadingColor};font-size:1.2rem;z-index:100;">
            <div style="width:60px;height:60px;border:4px solid rgba(147,51,234,0.2);border-top:4px solid ${loadingColor};border-radius:50%;animation:spin 1s linear infinite;margin-bottom:20px;"></div>
            <div style="font-weight:600;text-align:center;">${message}</div>
            ${searchType === 'semantic' ? '<div style="font-size:.9rem;color:#ccc;margin-top:10px;text-align:center;">AIが画像の内容を理解して検索しています...</div>' : ''}
        </div>
        <style>
            @keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
        </style>
    `;
    const galleryContainer = document.querySelector('.gallery-container');
    galleryContainer.style.position = 'relative';
}

export function displaySearchResults(results, searchType) {
    const grid = document.getElementById('mediaGrid');
    const statsInfo = document.getElementById('statsInfo');

    if (results.length === 0) {
        statsInfo.textContent = '検索結果: 0件';
        grid.innerHTML = '<div style="padding: 20px; color: #999;">検索結果がありません</div>';
        return;
    }

    const searchTypeLabel = searchType === 'filename' ? 'ファイル名' :
                            searchType === 'database' ? 'データベース' :
                            searchType === 'fulltext' ? '全文検索' : 'セマンティック検索';
    statsInfo.textContent = `検索結果: ${results.length}件 (${searchTypeLabel})`;
    grid.innerHTML = '';

    results.forEach(result => {
        const item = document.createElement('div');
        item.className = 'media-item';

        if (searchType === 'filename') {
            // allMediaFiles 由来のフルオブジェクトが来るので、ギャラリーと同じ生成ロジックを再利用
            const node = createSafeMediaItem(result, { showMeta: true });
            // click/contextmenu は下の共通ロジックで付与するため、ここでは node を item に差し替え
            // ただし createSafeMediaItem は .media-item を返すので、それをそのまま使う
            grid.appendChild(node);
            return; // 既に追加済みのため次へ
        } else if (searchType === 'database' || searchType === 'fulltext' || searchType === 'semantic') {
            const fileName = result.path.split('/').pop();
            item.dataset.path = result.path;

            // 媒体タイプ判定は拡張子ベースに限定（タグ/DBメタには依存しない）
            const isVideo = /\.(mp4|avi|mov|mkv|flv|wmv|webm|m4v|mpg|mpeg)$/i.test(fileName);
            const isAudio = /\.(mp3|wav|ogg|flac|m4a)$/i.test(fileName);
            const isModel = /\.(glb)$/i.test(fileName);

            const tags = Array.isArray(result.tags) ? result.tags.join(', ') : result.tags;

            let scoreDisplay = '';
            if (searchType === 'semantic') {
                let scoreText = '';
                if (result.rerank_score !== undefined) {
                    scoreText = `🧠 Rerank: ${(result.rerank_score * 100).toFixed(1)}% | 🎯 Vector: ${(result.score * 100).toFixed(1)}%`;
                } else if (result.score !== undefined) {
                    scoreText = `🎯 類似度: ${(result.score * 100).toFixed(1)}%`;
                }
                if (scoreText) {
                    scoreDisplay = `<div class=\"similarity-score\" style=\"font-size:.75rem;color:#9333ea;background:rgba(147,51,234,0.1);padding:2px 6px;border-radius:4px;display:inline-block;margin-top:4px;font-weight:600;\">${scoreText}<\/div>`;
                }
            }

            if (isVideo) {
                item.className = 'media-item video';
                item.dataset.type = 'video';
                item.innerHTML = sanitizeHtml(`
                    <div style=\"position: relative;\">\n  <video class=\"media-thumbnail\" src=\"${escapeHtml(result.path)}\" muted preload=\"metadata\"><\/video>\n  <button class=\"analyze-button\">解析<\/button>\n<\/div>\n<div class=\"media-info\">\n  <div class=\"media-name\">${escapeHtml(fileName)}<\/div>\n  <div class=\"media-details\" style=\"font-size:.75rem;color:#999;\">${escapeHtml(result.caption ? result.caption.substring(0, 50) + '...' : 'キャプションなし')}<\/div>\n  <div class=\"media-details\" style=\"font-size:.7rem;color:#666;\">${escapeHtml(tags || 'タグなし')}<\/div>\n  ${sanitizeHtml(scoreDisplay, { ALLOWED_TAGS: ['div'], ALLOWED_ATTR: ['style'] })}\n<\/div>
                `, {
                    ALLOWED_TAGS: ['div', 'video', 'button'],
                    ALLOWED_ATTR: ['class', 'style', 'src', 'muted', 'preload', 'onclick']
                });
            } else if (isAudio) {
                item.className = 'media-item audio';
                item.dataset.type = 'audio';
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
                    <div class=\"media-thumbnail audio\" style=\"background: ${escapeHtml(randomColor)};\">\n  <div class=\"audio-icon-wrapper\">\n    <span class=\"main-icon\">${escapeHtml(randomIcon)}<\/span>\n  <\/div>\n  <div class=\"audio-wave-bars\">\n    <span><\/span><span><\/span><span><\/span><span><\/span><span><\/span><span><\/span><span><\/span>\n  <\/div>\n<\/div>\n<button class=\"analyze-button\">解析<\/button>\n<div class=\"media-info\">\n  <div class=\"media-name\">${escapeHtml(fileName)}<\/div>\n  <div class=\"media-details\" style=\"font-size:.75rem;color:#999;\">${escapeHtml(result.caption ? result.caption.substring(0, 50) + '...' : 'キャプションなし')}<\/div>\n  <div class=\"media-details\" style=\"font-size:.7rem;color:#666;\">${escapeHtml(tags || 'タグなし')}<\/div>\n  ${sanitizeHtml(scoreDisplay, { ALLOWED_TAGS: ['div'], ALLOWED_ATTR: ['style'] })}\n<\/div>
                `, {
                    ALLOWED_TAGS: ['div', 'span', 'button'],
                    ALLOWED_ATTR: ['class', 'style']
                });
            } else if (isModel) {
                item.className = 'media-item model';
                item.dataset.type = 'model';
                item.innerHTML = sanitizeHtml(`
                    <div class=\"media-thumbnail model\" style=\"display:flex;align-items:center;justify-content:center;font-size:42px;color:#888;overflow:hidden;background:#111;\">\n  <div class=\"model-label\">🧊 3D<\/div>\n<\/div>\n<div class=\"media-info\">\n  <div class=\"media-name\">${escapeHtml(fileName)}<\/div>\n  <div class=\"media-details\" style=\"font-size:.75rem;color:#999;\">${escapeHtml(result.caption ? result.caption.substring(0, 50) + '...' : 'キャプションなし')}<\/div>\n  <div class=\"media-details\" style=\"font-size:.7rem;color:#666;\">${escapeHtml(tags || 'タグなし')}<\/div>\n  ${sanitizeHtml(scoreDisplay, { ALLOWED_TAGS: ['div'], ALLOWED_ATTR: ['style'] })}\n<\/div>
                `, { ALLOWED_TAGS: ['div'], ALLOWED_ATTR: ['class', 'style'] });

                const thumb = item.querySelector('.media-thumbnail.model');
                // スナップショット機能: 初回はアイコン、解析やモーダルで生成 → 次回以降は画像表示
                const snapshotBase = `.snapshots/${result.path.replace(/[\/\\]/g, '_')}`;
                const snapshotWebp = `${snapshotBase}.webp`;
                const snapshotPng = `${snapshotBase}.png`;
                let hasSnapshot = false;
                let activeSnapshotPath = null;
                
                // スナップショット存在確認
                async function checkSnapshot() {
                    try {
                        let response = await fetch(snapshotWebp, { method: 'HEAD' });
                        if (response.status === 200) { activeSnapshotPath = snapshotWebp; return true; }
                        response = await fetch(snapshotPng, { method: 'HEAD' });
                        if (response.status === 200) { activeSnapshotPath = snapshotPng; return true; }
                        activeSnapshotPath = null;
                        return false;
                    } catch {
                        return false;
                    }
                }
                
                // スナップショット生成
                async function generateSnapshot(modelViewer) {
                    console.log('🎯 generateSnapshot called for:', result.path);
                    try {
                        // model-viewerが読み込み完了まで待機
                        console.log('🔄 Waiting for model-viewer to load...');
                        await new Promise((resolve) => {
                            if (modelViewer.loaded) {
                                console.log('✅ Model already loaded');
                                resolve();
                            } else {
                                console.log('⏳ Waiting for load event...');
                                const timeout = setTimeout(() => {
                                    console.log('⚠️ Load timeout, proceeding anyway');
                                    resolve();
                                }, 10000);
                                modelViewer.addEventListener('load', () => {
                                    clearTimeout(timeout);
                                    console.log('✅ Model loaded via event');
                                    resolve();
                                }, { once: true });
                            }
                        });
                        
                        // 少し待ってから撮影（レンダリング安定化）
                        console.log('⏱️ Waiting for rendering stability...');
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        
                        // スナップショット撮影
                        console.log('📸 Taking snapshot with toBlob...');
                        const bigBlob = await modelViewer.toBlob({ 
                            mimeType: 'image/png', 
                            qualityArgument: 0.8 
                        });

                        if (!bigBlob) {
                            throw new Error('toBlob returned null');
                        }
                        console.log('✅ Blob created, size:', bigBlob.size, 'bytes');

                        // downscale to WEBP smaller
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
                        const targetPath = smallBlob ? snapshotWebp : snapshotPng;
                        const formData = new FormData();
                        formData.append('snapshot', smallBlob || bigBlob, smallBlob ? 'snapshot.webp' : 'snapshot.png');
                        formData.append('targetPath', targetPath);

                        const response = await fetch('/api/save-snapshot', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const responseData = await response.json();
                            hasSnapshot = true;
                            activeSnapshotPath = targetPath;
                            console.log('✅ Snapshot saved successfully:', targetPath, responseData);
                            // グローバル通知（一覧などにも反映）
                            window.dispatchEvent(new CustomEvent('snapshot-saved', { detail: { path: result.path } }));
                            return true;
                        } else {
                            const errorText = await response.text();
                            console.error('❌ Server error:', response.status, errorText);
                            throw new Error(`Server error: ${response.status}`);
                        }
                    } catch (error) {
                        console.error('❌ Failed to generate snapshot:', error);
                        console.error('Stack trace:', error.stack);
                    }
                    return false;
                }
                // 初期化：スナップショット存在確認
                (async function initModelThumbnail() {
                    hasSnapshot = await checkSnapshot();
                    if (hasSnapshot) {
                        showSnapshotImage();
                    } else {
                        addAnalyzeButton();
                    }
                })();

                function showSnapshotImage() {
                    const img = document.createElement('img');
                    img.src = activeSnapshotPath || snapshotWebp;
                    img.style.width = '100%';
                    img.style.height = '100%';
                    img.style.objectFit = 'cover';
                    thumb.innerHTML = '';
                    thumb.appendChild(img);
                    thumb.style.position = 'relative';
                    addAnalyzeButton();
                }

                function addAnalyzeButton() {
                    if (!thumb.querySelector('.analyze-button')) {
                        const btn = document.createElement('button');
                        btn.className = 'analyze-button';
                        btn.textContent = '解析';
                        btn.style.position = 'absolute';
                        btn.style.right = '10px';
                        btn.style.bottom = '10px';
                        btn.style.zIndex = '5';
                        btn.addEventListener('click', async (ev) => {
                            ev.stopPropagation();
                            
                            console.log('🔬 Analyze button clicked, hasSnapshot:', hasSnapshot);
                            
                            // 解析時にスナップショット生成（無い場合は一時viewerで生成）
                            if (!hasSnapshot) {
                                const mediaItem = btn.closest('.media-item');
                                let overlay = null;
                                if (mediaItem) {
                                    overlay = document.createElement('div');
                                    overlay.className = 'analysis-overlay';
                                    overlay.innerHTML = '<div class="analysis-spinner"></div><div>スナップショット生成中...</div>';
                                    mediaItem.style.position = 'relative';
                                    mediaItem.appendChild(overlay);
                                }
                                btn.textContent = 'スナップショット生成中...';
                                btn.disabled = true;
                                console.log('📷 Starting snapshot generation from analyze button...');
                                const tempViewer = document.createElement('model-viewer');
                                tempViewer.style.width = '1024px';
                                tempViewer.style.height = '1024px';
                                tempViewer.style.position = 'fixed';
                                tempViewer.style.left = '0px';
                                tempViewer.style.top = '0px';
                                tempViewer.style.opacity = '0';
                                tempViewer.style.pointerEvents = 'none';
                                tempViewer.style.zIndex = '-1';
                                tempViewer.setAttribute('reveal', 'auto');
                                tempViewer.setAttribute('disable-zoom', '');
                                tempViewer.setAttribute('disable-pan', '');
                                tempViewer.setAttribute('environment-image', 'neutral');
                                tempViewer.setAttribute('auto-rotate', '');
                                tempViewer.src = result.path;
                                document.body.appendChild(tempViewer);
                                await new Promise((resolve) => {
                                    let to = setTimeout(resolve, 8000);
                                    tempViewer.addEventListener('load', () => { clearTimeout(to); resolve(); }, { once: true });
                                });
                                await new Promise(r => setTimeout(r, 600));
                                const success = await generateSnapshot(tempViewer);
                                if (tempViewer && tempViewer.parentElement) tempViewer.parentElement.removeChild(tempViewer);
                                btn.textContent = '解析';
                                btn.disabled = false;
                                // overlay は解析へ引き継ぐため、ここでは消さない
                                if (success) {
                                    console.log('✅ Snapshot generated, updating display');
                                    hasSnapshot = true;
                                    showSnapshotImage();
                                    window.dispatchEvent(new CustomEvent('snapshot-saved', { detail: { path: result.path } }));
                                } else {
                                    console.log('❌ Snapshot generation failed from analyze button');
                                }
                            }
                            
                            // 解析実行
                            console.log('🧠 Starting model analysis...');
                            import('./analysis.js').then(m => m.analyzeModelAppearance(result.path, ev.currentTarget));
                        });
                        thumb.style.position = 'relative';
                        thumb.appendChild(btn);
                    }
                }
                
                // クリック時の拡大表示でもスナップショット生成
                item.addEventListener('click', async (ev) => {
                    if (ev.target.classList.contains('analyze-button')) return; // 解析ボタンは除外
                    
                    console.log('🖱️ GLB item clicked for expand view, hasSnapshot:', hasSnapshot);
                    
                    // 拡大表示時にスナップショット生成
                    if (!hasSnapshot) {
                        // 一時viewerでスナップショット生成
                        const tempViewer = document.createElement('model-viewer');
                        tempViewer.style.width = '1024px';
                        tempViewer.style.height = '1024px';
                        tempViewer.style.position = 'fixed';
                        tempViewer.style.left = '0px';
                        tempViewer.style.top = '0px';
                        tempViewer.style.opacity = '0';
                        tempViewer.style.pointerEvents = 'none';
                        tempViewer.style.zIndex = '-1';
                        tempViewer.setAttribute('reveal', 'auto');
                        tempViewer.setAttribute('disable-zoom', '');
                        tempViewer.setAttribute('disable-pan', '');
                        tempViewer.setAttribute('environment-image', 'neutral');
                        tempViewer.setAttribute('auto-rotate', '');
                        tempViewer.src = result.path;
                        document.body.appendChild(tempViewer);
                        await new Promise((resolve) => {
                            let to = setTimeout(resolve, 8000);
                            tempViewer.addEventListener('load', () => { clearTimeout(to); resolve(); }, { once: true });
                        });
                        await new Promise(r => setTimeout(r, 600));
                        const success = await generateSnapshot(tempViewer);
                        if (tempViewer && tempViewer.parentElement) tempViewer.parentElement.removeChild(tempViewer);
                        if (success) {
                            console.log('✅ Snapshot generated, updating display');
                            showSnapshotImage();
                        } else {
                            console.log('❌ Snapshot generation failed');
                        }
                    } else {
                        console.log('📸 Snapshot already exists, skipping generation');
                    }
                });
            } else {
                item.className = 'media-item image';
                item.dataset.type = 'image';
                item.innerHTML = sanitizeHtml(`
                    <img class=\"media-thumbnail\" src=\"${escapeHtml(result.path)}\" alt=\"${escapeHtml(fileName)}\" loading=\"lazy\">\n<button class=\"analyze-button\">解析<\/button>\n<div class=\"media-info\">\n  <div class=\"media-name\">${escapeHtml(fileName)}<\/div>\n  <div class=\"media-details\" style=\"font-size:.75rem;color:#999;\">${escapeHtml(result.caption ? result.caption.substring(0, 50) + '...' : 'キャプションなし')}<\/div>\n  <div class=\"media-details\" style=\"font-size:.7rem;color:#666;\">${escapeHtml(tags || 'タグなし')}<\/div>\n  ${sanitizeHtml(scoreDisplay, { ALLOWED_TAGS: ['div'], ALLOWED_ATTR: ['style'] })}\n<\/div>
                `, {
                    ALLOWED_TAGS: ['img', 'button', 'div'],
                    ALLOWED_ATTR: ['class', 'src', 'alt', 'loading', 'onclick', 'style']
                });
            }
        }

        item.addEventListener('click', () => {
            const fileName = result.path.split('/').pop();
            const fileType = item.dataset.type || 'image';
            const fileObj = { path: result.path, name: fileName, type: fileType };
            if (result.caption) fileObj.caption = result.caption;
            if (result.tags) fileObj.tags = result.tags;
            setSelectedFile(fileObj);
            openLightbox(fileObj);
        });

        item.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            const fileName = result.path.split('/').pop();
            const fileType = item.dataset.type || 'image';
            setSelectedFile({ path: result.path, name: fileName, type: fileType });
            showContextMenu(e.pageX, e.pageY);
        });

        // Attach analyze button listeners
        const btn = item.querySelector('.analyze-button');
        if (btn) {
            btn.addEventListener('click', (ev) => {
                ev.stopPropagation();
                if (item.dataset.type === 'video') {
                    import('./analysis.js').then(m => m.analyzeVideo(result.path, ev.currentTarget));
                } else if (item.dataset.type === 'audio') {
                    import('./analysis.js').then(m => m.analyzeAudio(result.path, ev.currentTarget));
                } else if (item.dataset.type === 'model') {
                    import('./analysis.js').then(m => m.analyzeModelAppearance(result.path, ev.currentTarget));
                } else {
                    import('./analysis.js').then(m => m.analyzeImage(result.path, ev.currentTarget));
                }
            });
        }

        if (item.dataset.type === 'video') {
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

        grid.appendChild(item);
    });
}

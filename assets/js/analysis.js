// 解析関連（ESM）
import { apiBaseUrl } from './api.js';
export { showAnalysisModal, closeAnalysisModal };

// 画像解析
export async function analyzeImage(filePath, buttonEl = null) {
    const ev = (typeof event !== 'undefined') ? event : null;
    const button = buttonEl || (ev && (ev.currentTarget || ev.target)) || null;
    const mediaItem = button ? button.closest('.media-item') : null;
    console.log('[AnalyzeImage] start', { filePath, apiBaseUrl });

    // 解析開始UI
    button.disabled = true;
    button.textContent = '解析中...';
    button.style.background = '#ff6b35';
    button.style.animation = 'pulse 1s infinite';

    // メディアアイテム全体にオーバーレイ表示
    const overlay = document.createElement('div');
    overlay.className = 'analysis-overlay';
    overlay.innerHTML = '<div class="analysis-spinner"></div><div>画像解析中...</div>';
    mediaItem.style.position = 'relative';
    mediaItem.appendChild(overlay);

    async function fallbackSnapshotAnalyze() {
        try {
            // ブラウザでラスタライズ（SVGなど非対応画像向け）
            const img = new Image();
            img.crossOrigin = 'anonymous';
            const done = new Promise((res, rej) => {
                img.onload = () => res();
                img.onerror = (e) => rej(e);
            });
            img.src = filePath;
            await done;
            const w = Math.min(2048, img.naturalWidth || img.width || 1024) || 1024;
            const h = Math.min(2048, img.naturalHeight || img.height || 1024) || 1024;
            const canvas = document.createElement('canvas');
            canvas.width = w; canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);
            let dataUrl = '';
            try { dataUrl = canvas.toDataURL('image/png'); } catch (_) { /* ignore */ }
            if (!dataUrl) throw new Error('snapshot-failed');
            const resp = await fetch(`${apiBaseUrl}/api/analyze-image-data`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageData: dataUrl, targetPath: filePath })
            });
            const data = await resp.json().catch(() => ({}));
            if (data && data.success && data.result) {
                showAnalysisModal(data.result);
                const mediaItem = button.closest('.media-item');
                if (mediaItem) { mediaItem.classList.add('highlighted'); setTimeout(() => mediaItem.classList.remove('highlighted'), 3000); }
                return true;
            }
            return false;
        } catch (e) { return false; }
    }

    try {
        const response = await fetch(`${apiBaseUrl}/api/analyze-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filePath })
        });

        const result = await response.json();
        console.log('[DEBUG] Server response:', result);

        if (result.success) {
            if (result.result && result.result.success) {
                console.log('[DEBUG] Analysis result:', result.result);
                // モーダルで結果を表示
                showAnalysisModal(result.result);
                // 解析成功時は画像をハイライト
                const mediaItem = button.closest('.media-item');
                if (mediaItem) {
                    mediaItem.classList.add('highlighted');
                    setTimeout(() => mediaItem.classList.remove('highlighted'), 3000);
                }
            } else {
                // Fallback: クライアントでラスタライズして画像データ解析
                const ok = await fallbackSnapshotAnalyze();
                if (!ok) {
                    alert(`解析は完了しましたが、結果が取得できませんでした。\n\nメッセージ: ${result.message}`);
                }
            }
        } else {
            alert(`解析に失敗しました: ${result.error || result.message}`);
        }
    } catch (error) {
        console.error('Analysis error:', error);
        // Fallback試行
        const ok = await fallbackSnapshotAnalyze();
        if (!ok) alert('解析中にエラーが発生しました');
    } finally {
        // UI復元
        button.disabled = false;
        button.textContent = '解析';
        button.style.background = '';
        button.style.animation = '';

        // オーバーレイを削除
        const overlay = mediaItem.querySelector('.analysis-overlay');
        if (overlay) overlay.remove();
    }
}

// 動画解析
export async function analyzeVideo(filePath, buttonEl = null) {
    const ev = (typeof event !== 'undefined') ? event : null;
    const button = buttonEl || (ev && (ev.currentTarget || ev.target)) || null;
    const mediaItem = button ? button.closest('.media-item') : null;
    console.log('[AnalyzeVideo] start', { filePath, apiBaseUrl });

    // 解析開始UI
    button.disabled = true;
    button.textContent = '解析中...';
    button.style.background = '#ff6b35';
    button.style.animation = 'pulse 1s infinite';

    // メディアアイテム全体にオーバーレイ表示
    const overlay = document.createElement('div');
    overlay.className = 'analysis-overlay';
    overlay.innerHTML = '<div class="analysis-spinner"></div><div>動画解析中...</div>';
    mediaItem.style.position = 'relative';
    mediaItem.appendChild(overlay);

    try {
        const response = await fetch(`${apiBaseUrl}/api/analyze-video`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filePath })
        });

        const result = await response.json();
        console.log('[DEBUG] Video analysis response:', result);

        if (result.success) {
            if (result.result) {
                console.log('[DEBUG] Video analysis result:', result.result);
                // モーダルで結果を表示
                if (result.result.caption || result.result.tags) {
                    showAnalysisModal(result.result);
                }
                // 解析成功時は動画をハイライト
                const mediaItem = button.closest('.media-item');
                if (mediaItem) {
                    mediaItem.classList.add('highlighted');
                    setTimeout(() => mediaItem.classList.remove('highlighted'), 3000);
                }
            } else {
                alert(`動画解析は完了しましたが、結果が取得できませんでした。\n\nメッセージ: ${result.message}`);
            }
        } else {
            const detail = result && result.details ? `\n詳細: ${result.details}` : '';
            alert(`動画解析に失敗しました: ${result.error || result.message || 'unknown'}${detail}`);
        }
    } catch (error) {
        console.error('Video analysis error:', error);
        alert('動画解析中にエラーが発生しました');
    } finally {
        // UI復元
        button.disabled = false;
        button.textContent = '解析';
        button.style.background = '';
        button.style.animation = '';

        // オーバーレイを削除
        const overlay = mediaItem.querySelector('.analysis-overlay');
        if (overlay) overlay.remove();
    }
}

// 音声解析
export async function analyzeAudio(filePath, buttonEl = null) {
    const ev = (typeof event !== 'undefined') ? event : null;
    const button = buttonEl || (ev && (ev.currentTarget || ev.target)) || null;
    const mediaItem = button ? button.closest('.media-item') : null;
    console.log('[AnalyzeAudio] start', { filePath, apiBaseUrl });

    // 解析開始UI
    button.disabled = true;
    button.textContent = '解析中...';
    button.style.background = '#ff6b35';
    button.style.animation = 'pulse 1s infinite';

    // メディアアイテム全体にオーバーレイ表示
    const overlay = document.createElement('div');
    overlay.className = 'analysis-overlay';
    overlay.innerHTML = '<div class="analysis-spinner"></div><div>音声解析中...</div>';
    mediaItem.style.position = 'relative';
    mediaItem.appendChild(overlay);

    try {
        const response = await fetch(`${apiBaseUrl}/api/analyze-audio`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filePath })
        });

        const result = await response.json();
        console.log('[DEBUG] Audio analysis response:', result);

        if (result.success) {
            if (result.result) {
                console.log('[DEBUG] Audio analysis result:', result.result);
                // モーダルで結果を表示
                if (result.result.caption || result.result.tags) {
                    showAnalysisModal(result.result);
                }
                // 解析成功時は音声をハイライト
                const mediaItem = button.closest('.media-item');
                if (mediaItem) {
                    mediaItem.classList.add('highlighted');
                    setTimeout(() => mediaItem.classList.remove('highlighted'), 3000);
                }
            } else {
                alert(`音声解析は完了しましたが、結果が取得できませんでした。\n\nメッセージ: ${result.message}`);
            }
        } else {
            alert(`音声解析に失敗しました: ${result.error || result.message}`);
        }
    } catch (error) {
        console.error('Audio analysis error:', error);
        alert('音声解析中にエラーが発生しました');
    } finally {
        // UI復元
        button.disabled = false;
        button.textContent = '解析';
        button.style.background = '';
        button.style.animation = '';

        // オーバーレイを削除
        const overlay = mediaItem.querySelector('.analysis-overlay');
        if (overlay) overlay.remove();
    }
}

// 3Dモデル解析
export async function analyzeModel(filePath) {
    const button = event.target;
    const mediaItem = button.closest('.media-item');

    // 解析開始UI
    button.disabled = true;
    button.textContent = '解析中...';
    button.style.background = '#ff6b35';
    button.style.animation = 'pulse 1s infinite';

    const overlay = document.createElement('div');
    overlay.className = 'analysis-overlay';
    overlay.innerHTML = '<div class="analysis-spinner"></div><div>3D解析中...</div>';
    mediaItem.style.position = 'relative';
    mediaItem.appendChild(overlay);

    try {
        const response = await fetch(`${apiBaseUrl}/api/analyze-model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filePath })
        });
        const result = await response.json();
        if (result.success && result.result && result.result.success) {
            showAnalysisModal(result.result);
            mediaItem.classList.add('highlighted');
            setTimeout(() => mediaItem.classList.remove('highlighted'), 2000);
        } else {
            alert(`3D解析に失敗しました: ${(result.result && result.result.error) || result.error || 'unknown'}`);
        }
    } catch (e) {
        console.error('Model analysis error:', e);
        alert('3D解析中にエラーが発生しました');
    } finally {
        button.disabled = false;
        button.textContent = '解析';
        button.style.background = '';
        button.style.animation = '';
        const overlayEl = mediaItem.querySelector('.analysis-overlay');
        if (overlayEl) overlayEl.remove();
    }
}

// 3Dモデル解析（ブラウザでスナップショット→サーバで画像解析）
export async function analyzeModelAppearance(filePath, buttonEl = null, mvEl = null) {
    const ev = (typeof event !== 'undefined') ? event : null;
    const button = buttonEl || (ev && ev.currentTarget) || (ev && ev.target) || null;
    let mediaItem = button ? button.closest('.media-item') : null;
    // ボタン経由で取得できない場合は data-path からカードを特定
    if (!mediaItem) {
        try {
            const sel = `.media-item[data-path="${CSS && CSS.escape ? CSS.escape(filePath) : filePath}"]`;
            mediaItem = document.querySelector(sel);
        } catch (_) {
            // CSS.escape が無い環境向けフォールバック
            const items = document.querySelectorAll('.media-item');
            mediaItem = Array.from(items).find(it => it.dataset && it.dataset.path === filePath) || null;
        }
    }
    // 常に独立した一時viewerでスナップショット（共有viewerの取り違え防止）
    let mv = null;
    let createdTemp = false;

    // UI開始
    console.log('[ModelAnalyze] start', { filePath });
    if (button) {
        button.disabled = true;
        button.textContent = '解析中...';
        button.style.background = '#ff6b35';
        button.style.animation = 'pulse 1s infinite';
    }
    // 既存のスナップショット用オーバーレイがあれば再利用して連続表示
    let overlay = null;
    let createdOverlay = false;
    if (mediaItem) {
        overlay = mediaItem.querySelector('.analysis-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'analysis-overlay';
            createdOverlay = true;
            mediaItem.style.position = 'relative';
            mediaItem.appendChild(overlay);
        }
        overlay.innerHTML = '<div class="analysis-spinner"></div><div>解析中...</div>';
    }

    try {
        mv = document.createElement('model-viewer');
        createdTemp = true;
        mv.style.width = '1024px';
        mv.style.height = '1024px';
        mv.style.position = 'fixed';
        mv.style.left = '0px';
        mv.style.top = '0px';
        mv.style.opacity = '0';
        mv.style.pointerEvents = 'none';
        mv.style.zIndex = '-1';
        mv.setAttribute('exposure', '1.0');
        mv.setAttribute('reveal', 'auto');
        mv.setAttribute('disable-zoom', '');
        mv.setAttribute('disable-pan', '');
        // 中立環境と落ち着いた見た目
        mv.setAttribute('environment-image', 'neutral');
        mv.setAttribute('auto-rotate', ''); // 初期表示でフレームを整えるため有効化
        // カメラはデフォルトの自動フレーミングに任せる
        mv.src = filePath;
        document.body.appendChild(mv);
        console.log('[ModelAnalyze] created temp viewer');

        // load を待つ
        if (createdTemp) {
            await new Promise((resolve) => {
                let timer = null;
                const onLoad = () => { if (timer) clearTimeout(timer); mv.removeEventListener('load', onLoad); resolve(); };
                mv.addEventListener('load', onLoad, { once: true });
                // safety timeout
                timer = setTimeout(resolve, 5000);
            });
        }
        // Lit の更新完了を待機（存在する場合）
        if (mv.updateComplete && typeof mv.updateComplete.then === 'function') {
            try { await mv.updateComplete; } catch {}
        }
        // レンダリング完了までキャンバスサイズをポーリング
        let tries = 0;
        while (tries < 60) {
            await new Promise(r => requestAnimationFrame(r));
            const canvasProbe = mv.shadowRoot && (mv.shadowRoot.getElementById('webgl-canvas') || mv.shadowRoot.querySelector('canvas'));
            if (canvasProbe && canvasProbe.width > 0 && canvasProbe.height > 0) break;
            tries++;
        }
        console.log('[ModelAnalyze] canvas probe tries:', tries);
        // 追加で数フレーム待って初期描画を安定させる
        for (let i = 0; i < 10; i++) {
            await new Promise(r => requestAnimationFrame(r));
        }

        // シャドウDOMからキャンバスを取得してデータURL化
        let dataUrl = '';
        try {
            const canvas = mv.shadowRoot && (mv.shadowRoot.getElementById('webgl-canvas') || mv.shadowRoot.querySelector('canvas'));
            if (canvas) {
                try {
                    dataUrl = canvas.toDataURL('image/png');
                    console.log('[ModelAnalyze] snapshot size:', dataUrl ? dataUrl.length : 0);
                } catch (err) {
                    console.warn('toDataURL failed:', err);
                }
            }
        } catch (e) {
            console.warn('Snapshot failed:', e);
        }

        if (!dataUrl) throw new Error('snapshot-failed');

        const response = await fetch(`${apiBaseUrl}/api/analyze-image-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageData: dataUrl, targetPath: filePath })
        });
        const result = await response.json().catch((e) => { console.warn('[ModelAnalyze] JSON parse failed', e); return {}; });
        console.log('[ModelAnalyze] server response', { status: response.status, ok: response.ok, body: result });
        if (result && result.success) {
            if (result.result) {
                try { 
                    showAnalysisModal(result.result);
                    // 解析モーダル表示直後にローディングを消す
                    if (overlay && overlay.parentElement) overlay.parentElement.removeChild(overlay);
                } catch {}
            } else {
                try { 
                    showAnalysisModal({ caption: '', tags: [], model: '' });
                    if (overlay && overlay.parentElement) overlay.parentElement.removeChild(overlay);
                } catch {}
            }
            if (mediaItem) {
                mediaItem.classList.add('highlighted');
                setTimeout(() => mediaItem.classList.remove('highlighted'), 2000);
            }
        } else {
            const msg = result && (result.error || result.details) ? `${result.error || ''} ${result.details || ''}`.trim() : `解析に失敗しました (HTTP ${response.status})`;
            alert(msg);
            return;
        }
    } catch (e) {
        console.error('Analysis error:', e);
        // スナップショットに失敗したらフォールバックでモデル解析を実行
        try {
            console.log('[ModelAnalyze] fallback /api/analyze-model');
            const resp = await fetch(`${apiBaseUrl}/api/analyze-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filePath })
            });
            const data = await resp.json().catch(() => ({}));
            console.log('[ModelAnalyze] fallback response', { status: resp.status, ok: resp.ok, body: data });
            if (data && data.success) {
                try { 
                    const meta = data.result || { caption: '', tags: [], model: '' };
                    showAnalysisModal(meta);
                    if (overlay && overlay.parentElement) overlay.parentElement.removeChild(overlay);
                } catch {}
            } else {
                alert('解析に失敗しました');
            }
        } catch (ee) {
            console.error('Fallback analyze-model failed:', ee);
            alert('解析に失敗しました');
        }
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = '解析';
            button.style.background = '';
            button.style.animation = '';
        }
        if (mediaItem) {
        if (overlay && overlay.parentElement) overlay.parentElement.removeChild(overlay);
        }
        // 後始末（臨時作成したviewerのみ破棄）
        if (createdTemp && mv && mv.parentElement) {
            mv.parentElement.removeChild(mv);
        }
    }
}

// モーダル表示
function showAnalysisModal(analysisData) {
    const modal = document.getElementById('analysisModal');
    const captionElement = document.getElementById('modalCaption');
    const tagsElement = document.getElementById('modalTags');
    const modelElement = document.getElementById('modalModel');

    // キャプション設定
    captionElement.textContent = analysisData.caption || '取得できませんでした';

    // タグ設定
    tagsElement.innerHTML = '';
    if (Array.isArray(analysisData.tags) && analysisData.tags.length > 0) {
        analysisData.tags.forEach(tag => {
            const tagElement = document.createElement('span');
            tagElement.className = 'modal-tag';
            tagElement.textContent = tag;
            tagsElement.appendChild(tagElement);
        });
    } else {
        const noTagsElement = document.createElement('div');
        noTagsElement.className = 'modal-value';
        noTagsElement.textContent = 'タグが見つかりませんでした';
        tagsElement.appendChild(noTagsElement);
    }

    // モデル設定
    modelElement.textContent = analysisData.model || '不明';

    // モーダル表示
    modal.classList.add('show');
    document.body.style.overflow = 'hidden';
}

function closeAnalysisModal() {
    const modal = document.getElementById('analysisModal');
    modal.classList.remove('show');
    document.body.style.overflow = 'auto';
}

// モーダル外クリック・ESCで閉じる
document.addEventListener('click', (e) => {
    const modal = document.getElementById('analysisModal');
    if (e.target === modal) {
        closeAnalysisModal();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAnalysisModal();
    }
});

// 全画像解析
export async function analyzeAllImages() {
    const analyzeBtn = document.getElementById('analyzeAllBtn');
    const grid = document.getElementById('mediaGrid');

    // 現在表示されている画像ファイルのみを取得
    const imageItems = grid.querySelectorAll('.media-item.image');
    if (imageItems.length === 0) {
        alert('解析する画像がありません');
        return;
    }

    if (!confirm(`${imageItems.length}枚の画像を解析します。続行しますか？`)) {
        return;
    }

    // ボタンを無効化
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '解析中...';

    // 進捗表示用の要素を作成
    const progressDiv = document.createElement('div');
    progressDiv.id = 'analysisProgress';
    progressDiv.style.cssText = `
        position: fixed; top: 20px; right: 20px; background: #1a1a1a; border: 1px solid #333;
        border-radius: 8px; padding: 15px; z-index: 1000; min-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,.5);
    `;
    document.body.appendChild(progressDiv);

    let completed = 0;
    let failed = 0;

    function updateProgress() {
        const total = imageItems.length;
        const progress = Math.round((completed + failed) / total * 100);
        progressDiv.innerHTML = `
            <div style="color: #4a9eff; font-weight: 600; margin-bottom: 10px;">画像解析進捗: ${progress}%</div>
            <div style=\"background: #333; border-radius: 4px; height: 8px; overflow: hidden;\">
                <div style=\"background: #4a9eff; height: 100%; width: ${progress}%; transition: width .3s;\"></div>
            </div>
            <div style="font-size: .85rem; color: #ccc; margin-top: 8px;">完了: ${completed} / 失敗: ${failed} / 残り: ${total - completed - failed}</div>
        `;
    }

    const CONCURRENT_LIMIT = 1; // 連続処理
    let processing = 0;
    let index = 0;

    const processNext = async () => {
        if (index >= imageItems.length) return;

        const item = imageItems[index];
        index++;
        processing++;

        try {
            const filePath = item.dataset.path;
            const response = await fetch(`${apiBaseUrl}/api/analyze-image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filePath })
            });
            const result = await response.json();

            if (result.success && result.result && result.result.success) {
                completed++;
                item.classList.add('highlighted');
                setTimeout(() => item.classList.remove('highlighted'), 1000);
            } else {
                failed++;
                item.style.border = '2px solid #ff6b6b';
                setTimeout(() => item.style.border = '', 3000);
            }
        } catch (error) {
            failed++;
            console.error('Analysis error:', error);
        } finally {
            processing--;
            updateProgress();

            if (index < imageItems.length && processing < CONCURRENT_LIMIT) {
                setTimeout(() => processNext(), 500);
            }

            if (completed + failed >= imageItems.length) {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '⚡ 画像全解析';
                setTimeout(() => {
                    document.body.removeChild(progressDiv);
                    alert(`解析完了！\n成功: ${completed}件\n失敗: ${failed}件`);
                }, 2000);
            }
        }
    };

    updateProgress();
    for (let i = 0; i < Math.min(CONCURRENT_LIMIT, imageItems.length); i++) {
        processNext();
    }
}

// 全動画解析
export async function analyzeAllVideos() {
    const analyzeBtn = document.getElementById('analyzeAllVideosBtn');
    const grid = document.getElementById('mediaGrid');

    // 現在表示されている動画ファイルのみを取得
    const videoItems = Array.from(grid.querySelectorAll('.media-item.video'));
    if (videoItems.length === 0) {
        alert('解析可能な動画ファイルが見つかりません。');
        return;
    }

    const CONCURRENT_LIMIT = 1; // 動画解析は重いため並列数を制限

    // ボタン無効化
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '🎬 解析中...';

    // 進捗表示を作成
    const progressDiv = document.createElement('div');
    progressDiv.style.cssText = `
        position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,.9); color: #fff; padding: 20px;
        border-radius: 8px; z-index: 9999; min-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,.3);
    `;
    document.body.appendChild(progressDiv);

    let currentIndex = 0;
    let completed = 0;
    let failed = 0;
    const total = videoItems.length;

    const updateProgress = () => {
        const processed = completed + failed;
        const percentage = Math.round((processed / total) * 100);
        progressDiv.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 10px;">🎬 動画全解析中...</div>
            <div style="margin-bottom: 5px;">進捗: ${processed}/${total} (${percentage}%)</div>
            <div style="margin-bottom: 5px;">✅ 成功: ${completed}件</div>
            <div style="margin-bottom: 10px;">❌ 失敗: ${failed}件</div>
            <div style="background: #2a2a2a; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: #4a9eff; height: 100%; width: ${percentage}%; transition: width .3s;"></div>
            </div>
        `;
        if (processed >= total) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🎬 動画全解析';
            setTimeout(() => {
                document.body.removeChild(progressDiv);
                alert(`動画解析完了！\n成功: ${completed}件\n失敗: ${failed}件`);
            }, 2000);
        }
    };

    const processNext = async () => {
        if (currentIndex >= videoItems.length) return;
        const videoItem = videoItems[currentIndex++];
        const videoPath = videoItem.getAttribute('data-path');
        try {
            const analyzeButton = videoItem.querySelector('.analyze-button');
            if (analyzeButton && !analyzeButton.disabled) {
                // オーバーレイ表示
                analyzeButton.disabled = true;
                analyzeButton.textContent = '解析中...';
                analyzeButton.style.background = '#ff6b35';
                analyzeButton.style.animation = 'pulse 1s infinite';

                const overlay = document.createElement('div');
                overlay.className = 'analysis-overlay';
                overlay.innerHTML = '<div class="analysis-spinner"></div><div>動画解析中...</div>';
                videoItem.style.position = 'relative';
                videoItem.appendChild(overlay);

                const response = await fetch(`${apiBaseUrl}/api/analyze-video`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filePath: videoPath })
                });
                const result = await response.json();

                if (result.success && result.result) {
                    completed++;
                    videoItem.classList.add('highlighted');
                    setTimeout(() => videoItem.classList.remove('highlighted'), 2000);
                } else {
                    failed++;
                }

                // UI復元
                analyzeButton.disabled = false;
                analyzeButton.textContent = '解析';
                analyzeButton.style.background = '';
                analyzeButton.style.animation = '';

                const overlayElement = videoItem.querySelector('.analysis-overlay');
                if (overlayElement) overlayElement.remove();
            } else {
                failed++;
            }
        } catch (error) {
            console.error('Video analysis error:', error);
            failed++;
            const analyzeButton = videoItem.querySelector('.analyze-button');
            if (analyzeButton) {
                analyzeButton.disabled = false;
                analyzeButton.textContent = '解析';
                analyzeButton.style.background = '';
                analyzeButton.style.animation = '';
            }
            const overlayElement = videoItem.querySelector('.analysis-overlay');
            if (overlayElement) overlayElement.remove();
        }
        updateProgress();
        if (currentIndex < videoItems.length) processNext();
    };

    // 初期進捗表示
    updateProgress();
    for (let i = 0; i < Math.min(CONCURRENT_LIMIT, videoItems.length); i++) {
        processNext();
    }
}

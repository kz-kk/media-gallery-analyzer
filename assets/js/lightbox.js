// ライトボックス機能（ESM）
import { escapeHtml } from './security.js';
import { apiBaseUrl } from './api.js';

// ライトボックスを開く
export function openLightbox(file) {
    const lightbox = document.getElementById('lightbox');
    const content = document.getElementById('lightboxContent');

    // Clear previous content
    content.innerHTML = '';

    // Build two-column layout: media (left) + meta (right)
    const mediaWrap = document.createElement('div');
    mediaWrap.className = 'lightbox-media';

    if (file.type === 'image') {
        const img = document.createElement('img');
        img.src = escapeHtml(file.path);
        img.alt = escapeHtml(file.name);
        mediaWrap.appendChild(img);
    } else if (file.type === 'video') {
        const video = document.createElement('video');
        video.src = escapeHtml(file.path);
        video.controls = true;
        video.autoplay = true;
        mediaWrap.appendChild(video);
    } else if (file.type === 'audio') {
        const container = document.createElement('div');
        container.style.textAlign = 'center';
        container.style.padding = '40px';

        const icon = document.createElement('div');
        icon.style.fontSize = '5rem';
        icon.style.marginBottom = '20px';
        icon.textContent = '🎵';

        const fileName = document.createElement('div');
        fileName.style.fontSize = '1.2rem';
        fileName.style.marginBottom = '20px';
        fileName.textContent = file.name;

        const audio = document.createElement('audio');
        audio.src = escapeHtml(file.path);
        audio.controls = true;
        audio.autoplay = true;
        audio.style.width = '300px';

        container.appendChild(icon);
        container.appendChild(fileName);
        container.appendChild(audio);
        mediaWrap.appendChild(container);
    } else if (file.type === 'model') {
        // 3Dモデル表示（model-viewer）
        const mv = document.createElement('model-viewer');
        mv.setAttribute('src', escapeHtml(file.path));
        mv.setAttribute('camera-controls', '');
        mv.setAttribute('disable-zoom', '');
        mv.setAttribute('auto-rotate', '');
        mv.style.width = '100%';
        mv.style.height = '100%';
        mv.style.minHeight = '420px';
        mediaWrap.appendChild(mv);

        // モーダル表示時にスナップショットが無ければ生成
        (async () => {
            const snapshotBase = `.snapshots/${file.path.replace(/[\\/\\]/g, '_')}`;
            const snapshotWebp = `${snapshotBase}.webp`;
            const snapshotPng = `${snapshotBase}.png`;
            let activeSnapshotPath = null;
            try {
                let head = await fetch(snapshotWebp, { method: 'HEAD' });
                if (head.status === 200) activeSnapshotPath = snapshotWebp;
                else {
                    head = await fetch(snapshotPng, { method: 'HEAD' });
                    if (head.status === 200) activeSnapshotPath = snapshotPng;
                }
            } catch (_) { activeSnapshotPath = null; }
            if (!activeSnapshotPath) {
                // load待機→安定化→toBlob→保存
                await new Promise((resolve) => {
                    if (mv.loaded) return resolve();
                    const to = setTimeout(resolve, 10000);
                    mv.addEventListener('load', () => { clearTimeout(to); resolve(); }, { once: true });
                });
                await new Promise(r => setTimeout(r, 800));
                try {
                    const bigBlob = await mv.toBlob({ mimeType: 'image/png', qualityArgument: 0.85 });
                    if (bigBlob) {
                        // downscale to WEBP small
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
                            // 一覧側のサムネを即時更新
                            const items = document.querySelectorAll('.media-item.model');
                            for (const it of items) {
                                if (it.dataset && it.dataset.path === file.path) {
                                    const th = it.querySelector('.media-thumbnail.model');
                                    if (th) {
                                        const img2 = new Image();
                                        img2.className = 'media-thumbnail';
                                        img2.style.objectFit = 'cover';
                                        img2.style.width = '100%';
                                        img2.style.height = '100%';
                                        img2.onload = () => { th.innerHTML = ''; th.appendChild(img2); };
                                        const usePath = smallBlob ? snapshotWebp : snapshotPng;
                                        img2.src = usePath + `?t=${Date.now()}`;
                                    }
                                }
                            }
                            // グローバル通知で他のビューにも反映させる
                            window.dispatchEvent(new CustomEvent('snapshot-saved', { detail: { path: file.path } }));
                        }
                    }
                } catch (_) { /* ignore snapshot errors */ }
            }
        })();
    }

    const metaWrap = document.createElement('div');
    metaWrap.className = 'lightbox-meta';

    const title = document.createElement('div');
    title.className = 'lightbox-meta-title';
    title.textContent = file.name || '詳細情報';
    metaWrap.appendChild(title);

    const captionLabel = document.createElement('div');
    captionLabel.className = 'lightbox-meta-label';
    captionLabel.textContent = 'キャプション';
    metaWrap.appendChild(captionLabel);

    const captionValue = document.createElement('div');
    captionValue.className = 'lightbox-meta-caption';
    const initialCaption = (file.caption && String(file.caption).trim()) ? String(file.caption) : '';
    captionValue.textContent = initialCaption || '—';
    metaWrap.appendChild(captionValue);

    const tagsLabel = document.createElement('div');
    tagsLabel.className = 'lightbox-meta-label';
    tagsLabel.textContent = 'タグ';
    metaWrap.appendChild(tagsLabel);

    const tagsBox = document.createElement('div');
    tagsBox.className = 'lightbox-meta-tags';
    const tags = Array.isArray(file.tags) ? file.tags
        : (typeof file.tags === 'string' ? file.tags.split(',').map(t => t.trim()).filter(Boolean) : []);
    if (tags.length > 0) {
        tags.forEach(t => {
            const chip = document.createElement('span');
            chip.className = 'meta-tag';
            chip.textContent = t;
            tagsBox.appendChild(chip);
        });
    } else {
        const none = document.createElement('div');
        none.className = 'lightbox-meta-empty';
        none.textContent = '—';
        tagsBox.appendChild(none);
    }
    metaWrap.appendChild(tagsBox);

    content.appendChild(mediaWrap);
    content.appendChild(metaWrap);

    lightbox.classList.add('active');

    // 補完: 検索前などメタ情報が無い場合はAPIから取得して更新
    // 常にバックグラウンドで最新メタを取りに行き、取得できたらUIを更新
    async function refreshMetaOnce() {
        if (!apiBaseUrl) return;
        try {
            const url = `${apiBaseUrl}/api/metadata?path=${encodeURIComponent(file.path)}`;
            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                if (data && data.success && data.media) {
                    const cap = data.media.caption;
                    const tg = data.media.tags;
                    if (cap && String(cap).trim()) {
                        captionValue.textContent = String(cap);
                    }
                    if (tg && (Array.isArray(tg) ? tg.length > 0 : String(tg).trim().length > 0)) {
                        tagsBox.innerHTML = '';
                        const list = Array.isArray(tg) ? tg : String(tg).split(',').map(s => s.trim()).filter(Boolean);
                        list.forEach(t => {
                            const chip = document.createElement('span');
                            chip.className = 'meta-tag';
                            chip.textContent = t;
                            tagsBox.appendChild(chip);
                        });
                    }
                }
            }
        } catch (_) { /* ignore */ }
    }
    refreshMetaOnce().catch(() => {});
}

// ライトボックスを閉じる
export function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    const content = document.getElementById('lightboxContent');
    lightbox.classList.remove('active');

    const media = content.querySelector('video, audio');
    if (media) {
        media.pause();
    }

    setTimeout(() => {
        content.innerHTML = '';
    }, 300);
}

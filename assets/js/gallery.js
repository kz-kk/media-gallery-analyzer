// ギャラリー描画・表示更新（ESM）
import { createSafeMediaItem } from './security.js';
import { openLightbox } from './lightbox.js';
import { showContextMenu } from './context-menu.js';
import { allMediaFiles, getCurrentFilter, setSelectedFile } from './state.js';

export function displayMedia(files = allMediaFiles) {
  const grid = document.getElementById('mediaGrid');

  // 既存の動画を停止してメモリ解放
  const oldVideos = grid.querySelectorAll('video');
  oldVideos.forEach(video => {
    video.pause();
    video.src = '';
    video.load();
  });

  grid.innerHTML = '';

  const fragment = document.createDocumentFragment();
  const maxInitialLoad = 50; // 初期表示数を制限
  const displayFiles = files.slice(0, maxInitialLoad);

  displayFiles.forEach(file => {
    const item = createSafeMediaItem(file);

    if (file.type === 'video') {
      const video = item.querySelector('video');
      if (video) {
        video.addEventListener('loadedmetadata', () => {
          video.currentTime = 0.1;
          const duration = video.duration;
          const minutes = Math.floor(duration / 60);
          const seconds = Math.floor(duration % 60);
          const durationEl = item.querySelector('.video-duration');
          if (durationEl) {
            durationEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
          }
        });
      }
    }

    item.addEventListener('click', () => openLightbox(file));
    item.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      setSelectedFile(file);
      showContextMenu(e.pageX, e.pageY);
    });

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
  });

  grid.appendChild(fragment);

  if (files.length > maxInitialLoad) {
    const container = document.createElement('div');
    container.className = 'load-more';
    const btn = document.createElement('button');
    btn.className = 'btn-load-more';
    const remaining = files.length - maxInitialLoad;
    btn.innerHTML = `<span class="label">さらに読み込む (${remaining} 件)</span>`;
    btn.addEventListener('click', () => {
      btn.classList.add('loading');
      import('./ui-handlers.js').then(m => m.loadMoreMedia(maxInitialLoad));
    });
    container.appendChild(btn);
    // const hint = document.createElement('div');
    // hint.className = 'load-more-hint';
    // hint.textContent = 'スクロールしても読み込めます';
    // container.appendChild(hint);
    grid.appendChild(container);
  }
}

export function filterByFolder(folderPath) {
  const filtered = allMediaFiles.filter(file => file.path.startsWith(folderPath));
  displayMedia(filtered);
  document.getElementById('statsInfo').textContent = `Showing ${filtered.length} files from ${folderPath}`;
}

export function filterByType(type) {
  // state は setter で保持されるが、ここでは描画のみ
  const files = type === 'all'
    ? allMediaFiles
    : allMediaFiles.filter(file => file.type === type);
  displayMedia(files);
  document.getElementById('statsInfo').textContent = (
    type === 'all'
      ? 'Showing all files'
      : `Showing ${files.length} ${type} files`
  );
}

export function updateStats() {
  const count = allMediaFiles.length;
  document.getElementById('fileCount').textContent = `${count} items`;

  const totalSize = allMediaFiles.reduce((sum, file) => sum + (file.size || 0), 0);
  const sizeInMB = (totalSize / (1024 * 1024)).toFixed(2);
  document.getElementById('totalSize').textContent = `Total: ${sizeInMB} MB`;
}

export function showFileInput() {
  const grid = document.getElementById('mediaGrid');
  grid.innerHTML = `
        <div class="error">
            <div>⚠️ Node.jsサーバーが起動していません</div>
            <div class="error-message">
                <p>動的スキャンを使用するには:</p>
                <ol style="text-align: left; margin-top: 10px;">
                    <li>ターミナルを開く</li>
                    <li>このディレクトリに移動</li>
                    <li>実行: <code>node server.js</code></li>
                    <li>ページをリロード</li>
                </ol>
            </div>
        </div>
    `;
}

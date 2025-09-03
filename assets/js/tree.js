// フォルダツリー構築（ESM）
import { clearAllMediaFiles, addMediaFile, allMediaFiles, setCurrentFolderPath } from './state.js';
import { displayMedia, updateStats } from './gallery.js';

export function getTotalFileCount(folder) {
  let totalCount = 0;
  if (folder.items && folder.items.files) {
    totalCount += folder.items.files.length;
  }
  if (folder.items && folder.items.folders) {
    folder.items.folders.forEach(subfolder => {
      totalCount += getTotalFileCount(subfolder);
    });
  }
  return totalCount;
}

export function createFolderItem(folder) {
  const container = document.createElement('div');
  container.className = 'tree-item';

  const label = document.createElement('div');
  label.className = 'tree-label';

  const toggle = document.createElement('span');
  toggle.className = 'tree-toggle';
  toggle.textContent = '▶';
  label.appendChild(toggle);

  const icon = document.createElement('span');
  icon.className = 'tree-icon';
  icon.textContent = '📁';
  label.appendChild(icon);

  const name = document.createElement('span');
  name.className = 'tree-name';
  name.textContent = folder.name;
  label.appendChild(name);

  const count = document.createElement('span');
  count.className = 'tree-count';
  count.textContent = getTotalFileCount(folder);
  label.appendChild(count);

  container.appendChild(label);

  const childrenContainer = document.createElement('div');
  childrenContainer.className = 'tree-children';

  if (folder.items && folder.items.files && folder.items.files.length > 0) {
    folder.items.files.forEach(file => {
      addMediaFile(file);
    });
  }
  if (folder.items && folder.items.folders && folder.items.folders.length > 0) {
    folder.items.folders.forEach(subfolder => {
      const subfolderItem = createFolderItem(subfolder);
      childrenContainer.appendChild(subfolderItem);
    });
  }

  container.appendChild(childrenContainer);

  label.addEventListener('click', (e) => {
    e.stopPropagation();
    childrenContainer.classList.toggle('expanded');
    toggle.classList.toggle('expanded');
    toggle.textContent = childrenContainer.classList.contains('expanded') ? '▼' : '▶';
    // 単一選択にするため既存の選択を解除して現在を選択
    const treeRoot = document.getElementById('folderTree');
    if (treeRoot) {
      treeRoot.querySelectorAll('.tree-label.selected').forEach(el => el.classList.remove('selected'));
    }
    label.classList.add('selected');

    const folderFiles = [];
    function collectFiles(folderData) {
      if (folderData.items && folderData.items.files) {
        folderData.items.files.forEach(file => folderFiles.push(file));
      }
      if (folderData.items && folderData.items.folders) {
        folderData.items.folders.forEach(subfolder => collectFiles(subfolder));
      }
    }
    collectFiles(folder);
    // 検索スコープ用に現在のフォルダパスを更新
    if (folder && folder.path) {
      setCurrentFolderPath(folder.path);
    }
    displayMedia(folderFiles);
    document.getElementById('statsInfo').textContent = `Showing ${folderFiles.length} files from ${folder.name}`;
  });

  return container;
}

export function processScannedData(data) {
  clearAllMediaFiles();
  const folderTree = document.getElementById('folderTree');
  folderTree.innerHTML = '';

  if (data.data.files && data.data.files.length > 0) {
    data.data.files.forEach(file => {
      addMediaFile(file);
    });
  }

  const rootContainer = document.createElement('div');
  rootContainer.className = 'tree-item';

  const rootLabel = document.createElement('div');
  rootLabel.className = 'tree-label selected';

  const rootToggle = document.createElement('span');
  rootToggle.className = 'tree-toggle expanded';
  rootToggle.textContent = '▼';
  rootLabel.appendChild(rootToggle);

  const rootIcon = document.createElement('span');
  rootIcon.className = 'tree-icon';
  rootIcon.textContent = '📁';
  rootLabel.appendChild(rootIcon);

  const rootName = document.createElement('span');
  rootName.className = 'tree-name';
  const dirName = data.basePath ? data.basePath.split('/').pop() || 'Media' : 'Media';
  rootName.textContent = dirName;
  rootLabel.appendChild(rootName);

  const rootCount = document.createElement('span');
  rootCount.className = 'tree-count';
  rootCount.textContent = '0';
  rootLabel.appendChild(rootCount);

  rootContainer.appendChild(rootLabel);

  const rootChildren = document.createElement('div');
  rootChildren.className = 'tree-children expanded';

  if (data.data.folders && data.data.folders.length > 0) {
    data.data.folders.forEach(folder => {
      const folderItem = createFolderItem(folder);
      rootChildren.appendChild(folderItem);
    });
  }

  rootContainer.appendChild(rootChildren);

  rootCount.textContent = allMediaFiles.length;

  rootLabel.addEventListener('click', () => {
    rootChildren.classList.toggle('expanded');
    rootToggle.classList.toggle('expanded');
    rootToggle.textContent = rootChildren.classList.contains('expanded') ? '▼' : '▶';
    // 単一選択にするため既存の選択を解除して現在を選択
    const treeRoot = document.getElementById('folderTree');
    if (treeRoot) {
      treeRoot.querySelectorAll('.tree-label.selected').forEach(el => el.classList.remove('selected'));
    }
    rootLabel.classList.add('selected');
    // ルート選択時は全体（=空文字）をスコープに
    setCurrentFolderPath('');
    displayMedia(allMediaFiles);
  });

  folderTree.appendChild(rootContainer);
  displayMedia(allMediaFiles);
  updateStats();
  // 初期状態はルートを選択扱い
  setCurrentFolderPath('');
}

// ESM shared state and helpers

export let allMediaFiles = [];
export function clearAllMediaFiles() {
  allMediaFiles.length = 0;
}
export function addMediaFile(file) {
  allMediaFiles.push(file);
}

export let currentFilter = 'all';
export function setCurrentFilter(v) { currentFilter = v; }
export function getCurrentFilter() { return currentFilter; }

export let currentPath = '';
export function setCurrentPath(v) { currentPath = v; }

let _selectedFile = null;
export function setSelectedFile(v) { _selectedFile = v; }
export function getSelectedFile() { return _selectedFile; }

// 現在選択しているフォルダ（ルートは空文字）
export let currentFolderPath = '';
export function setCurrentFolderPath(p) { currentFolderPath = p || ''; }
export function getCurrentFolderPath() { return currentFolderPath; }

// Optional debounce handle for search
let _searchTimeout;
export function setSearchTimeout(v) { _searchTimeout = v; }
export function getSearchTimeout() { return _searchTimeout; }

// Metadata cache keyed by relative path
const _metaCache = new Map();
export function setMetaForPath(path, meta) {
  if (!path) return;
  try {
    const prev = _metaCache.get(path) || {};
    _metaCache.set(path, { ...prev, ...meta });
  } catch (e) {}
}
export function getMetaForPath(path) {
  if (!path) return null;
  return _metaCache.get(path) || null;
}

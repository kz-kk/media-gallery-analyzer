// API関連（ESM）
import { processScannedData } from './tree.js';
import { showFileInput } from './gallery.js';
import { getSelectedFile } from './state.js';
import { hideContextMenu } from './context-menu.js';

// APIベースURL
export let apiBaseUrl = '';

// サーバー設定を取得
export async function getServerConfig() {
    // まず現在のポートで試す（HTMLが同じサーバーから配信されている場合）
    try {
        const currentPort = window.location.port || '80';
        const currentHost = window.location.hostname || '127.0.0.1';
        const response = await fetch(`http://${currentHost}:${currentPort}/api/config`);
        if (response.ok) {
            const config = await response.json();
            apiBaseUrl = config.serverBaseUrl;
            return true;
        }
    } catch (e) {
        // 現在のポートで失敗
    }
    
    // 一般的なポートを試す
    const commonPorts = [7777, 8080, 3000, 3333];
    for (const port of commonPorts) {
        try {
            const response = await fetch(`http://127.0.0.1:${port}/api/config`);
            if (response.ok) {
                const config = await response.json();
                apiBaseUrl = config.serverBaseUrl;
                return true;
            }
        } catch (e) {
            // このポートでは接続できない
        }
    }
    return false;
}

// ディレクトリをスキャン
export async function scanDirectory() {
    const folderTree = document.getElementById('folderTree');
    const grid = document.getElementById('mediaGrid');
    
    // ローディング表示
    folderTree.innerHTML = '<div class="loading">スキャン中...</div>';
    grid.innerHTML = '<div class="loading">メディアファイルを読み込み中...</div>';
    
    // サーバー設定を取得
    if (!apiBaseUrl) {
        const configSuccess = await getServerConfig();
        if (!configSuccess) {
            console.error('Could not connect to server');
            folderTree.innerHTML = '<div class="error">サーバーに接続できません</div>';
            showFileInput();
            return;
        }
    }
    
    try {
        // Node.jsサーバーから取得
        const response = await fetch(`${apiBaseUrl}/api/scan`);
        
        if (!response.ok) {
            throw new Error('Server response not ok');
        }
        
        const data = await response.json();
        
        console.log('Scanned data:', data);
        processScannedData(data);
    } catch (error) {
        console.error('Error scanning directory:', error);
        
        // サーバーが動いていない場合の代替処理
        folderTree.innerHTML = '<div class="error">サーバーエラー</div>';
        showFileInput();
    }
}

// ファイルを開く
export async function openFile() {
    const selectedFile = getSelectedFile();
    if (selectedFile) {
        // Node.jsサーバー経由で開く必要がある
        fetch(`${apiBaseUrl}/api/open-file`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: selectedFile.path })
        }).catch(err => {
            // サーバーに機能がない場合、新しいタブで開く
            window.open(selectedFile.path, '_blank');
        });
    }
    hideContextMenu();
}

// フォルダを開く
export async function openFolder() {
    const selectedFile = getSelectedFile();
    if (selectedFile) {
        const folderPath = selectedFile.path.substring(0, selectedFile.path.lastIndexOf('/'));
        fetch(`${apiBaseUrl}/api/open-folder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: folderPath || '.' })
        }).catch(err => {
            console.error('Cannot open folder:', err);
        });
    }
    hideContextMenu();
}

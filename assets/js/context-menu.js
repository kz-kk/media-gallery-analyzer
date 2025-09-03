// コンテキストメニュー機能（ESM）
import { getSelectedFile } from './state.js';

// コンテキストメニューを表示
export function showContextMenu(x, y) {
    const contextMenu = document.getElementById('contextMenu');
    contextMenu.style.left = x + 'px';
    contextMenu.style.top = y + 'px';
    contextMenu.classList.add('active');
}

// コンテキストメニューを隠す
export function hideContextMenu() {
    const contextMenu = document.getElementById('contextMenu');
    contextMenu.classList.remove('active');
}

// ファイルパスをコピー
export function copyFilePath() {
    const selectedFile = getSelectedFile();
    if (selectedFile) {
        // fullPathがあればそれを使用、なければpathを使用
        const fullPath = selectedFile.fullPath || selectedFile.path;
        navigator.clipboard.writeText(fullPath).then(() => {
            console.log('Copied:', fullPath);
            // 通知を表示（オプション）
            const notification = document.createElement('div');
            notification.textContent = 'パスをコピーしました';
            notification.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #4a9eff;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                z-index: 10000;
            `;
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 2000);
        });
    }
    hideContextMenu();
}

// openFile()とopenFolder()はapi.jsにすでに存在するため削除

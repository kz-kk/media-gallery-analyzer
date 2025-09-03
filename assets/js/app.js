// Application entry (ESM)
// 明示的に主要モジュールを読み込んだ上で初期化
import './state.js';
import './security.js';
import './lightbox.js';
import './context-menu.js';
import './gallery.js';
import './tree.js';
import './api.js';
import './search.js';
import './analysis.js';
import { initializeApp } from './ui-handlers.js';

window.addEventListener('DOMContentLoaded', initializeApp);

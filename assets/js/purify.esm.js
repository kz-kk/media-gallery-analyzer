// ESM adapter for DOMPurify using local UMD build
// Loads purify.min.js (UMD) and re-exports DOMPurify for module consumers
import './purify.min.js';

// In browsers, purify.min.js attaches DOMPurify to window
export const DOMPurify = window.DOMPurify;
export default window.DOMPurify;
export function sanitize(html, options) {
  return window.DOMPurify.sanitize(html, options);
}

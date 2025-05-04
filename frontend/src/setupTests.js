// Polyfill ResizeObserver for jsdom environment (fixes ReferenceError in React Flow/graph tests)
global.ResizeObserver = class {
  constructor(callback) {
    this.callback = callback;
    this.elements = new Set();
  }
  observe(element) {
    this.elements.add(element);
  }
  unobserve(element) {
    this.elements.delete(element);
  }
  disconnect() {
    this.elements.clear();
  }
};
// Verification log to ensure polyfill is present before any imports
// eslint-disable-next-line no-console
console.log('[Test Setup] ResizeObserver polyfill applied:', typeof global.ResizeObserver !== 'undefined');
// Polyfill BroadcastChannel for all tests (JSDOM does not support it natively)
if (typeof window !== 'undefined' && typeof window.BroadcastChannel === 'undefined') {
  class MockBroadcastChannel {
    constructor() {
      this.onmessage = null;
    }
    postMessage() {}
    close() {}
    addEventListener() {}
    removeEventListener() {}
  }
  window.BroadcastChannel = MockBroadcastChannel;
}
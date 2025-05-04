import { useCallback, useEffect, useRef, useState, useContext } from 'react';
import { useNavigate, useLocation, UNSAFE_NavigationContext } from 'react-router-dom';

/**
 * useSyncBeforeNavigation
 * 
 * Ensures graph data is fully synchronized before allowing navigation between tabs.
 * Blocks navigation until the provided async sync function completes.
 * 
 * @param {Object}   options
 * @param {boolean}  options.shouldBlock - Whether navigation should be blocked (e.g., unsynced changes).
 * @param {Function} options.syncFn - Async function to synchronize data. Should return true if sync succeeded.
 * @param {Array}    options.dependencies - Array of dependencies (graph data, tab state, etc.) to track.
 * @returns {{
 *   isBlocking: boolean,
 *   isSyncing: boolean,
 *   isSynced: boolean,
 *   syncError: any,
 *   requestNavigation: (targetPath: string) => void,
 *   status: 'idle'|'syncing'|'synced'|'error',
 *   debugLog: string[]
 * }}
 */
export function useSyncBeforeNavigation(options) {
  // Defensive: allow undefined/null or missing options, provide defaults
  const {
    shouldBlock = false,
    syncFn = null,
    dependencies = []
  } = options || {};

  // Verbose debug logging (only in non-production)
  const isDev = typeof process !== 'undefined' && process.env && process.env.NODE_ENV !== 'production';
  const initialOptionsLog = isDev
    ? `useSyncBeforeNavigation initialized with options: ${JSON.stringify({
        shouldBlock,
        hasSyncFn: typeof syncFn === 'function',
        dependenciesType: Array.isArray(dependencies) ? 'array' : typeof dependencies
      })}`
    : null;
// Debug log state: start with initial options log if in dev, else empty
  // Debug log ref: start with initial options log if in dev, else empty
  const debugLogRef = useRef(isDev && initialOptionsLog ? [initialOptionsLog] : []);
  // Always call hooks unconditionally (React rules)
  const navigate = useNavigate();
  const location = useLocation();
  const navigationContext = useContext(UNSAFE_NavigationContext);
  // Defensive: navigator may be undefined/null in some test environments
  const navigator = navigationContext && typeof navigationContext === 'object' ? navigationContext.navigator : null;

  // Status: 'idle' | 'syncing' | 'synced' | 'error'
  const [status, setStatus] = useState('idle');
  const [pendingPath, setPendingPath] = useState(null);
  const [syncError, setSyncError] = useState(null);
  const asyncOpRef = useRef(null);
  const unblockRef = useRef(null);

  // Defensive: null/undefined checks for shouldBlock, syncFn, and navigator
  const safeShouldBlock = !!shouldBlock && typeof syncFn === 'function' && !!navigator;

  // Debug logging utility (ref-based, does not cause re-render)
  const log = useCallback((msg) => {
    debugLogRef.current.push(`[${new Date().toISOString()}] ${msg}`);
  }, []);

  // Block navigation when shouldBlock is true
  useEffect(() => {
    if (!safeShouldBlock) {
      if (unblockRef.current) {
        unblockRef.current();
        unblockRef.current = null;
        log('Navigation unblocked (shouldBlock false or navigator missing)');
      }
      return;
    }

    // Block navigation
    // Defensive: navigator.block may not exist
    if (navigator && typeof navigator.block === 'function') {
      unblockRef.current = navigator.block((tx) => {
        const nextPath = tx && tx.location && typeof tx.location.pathname === 'string'
          ? tx.location.pathname
          : '';
        log(`Navigation attempt to "${nextPath}" blocked, starting sync...`);
        setPendingPath(nextPath);
        setStatus('syncing');
      });
      log('Navigation blocking enabled');
    } else {
      log('Navigator.block is not available; navigation blocking not enabled');
    }

    // Cleanup
    return () => {
      if (unblockRef.current) {
        unblockRef.current();
        unblockRef.current = null;
        log('Navigation blocking cleaned up');
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeShouldBlock, navigator, ...dependencies]);

  // Handle browser tab close (beforeunload)
  useEffect(() => {
    if (!safeShouldBlock) return;
    const handleBeforeUnload = (e) => {
      e.preventDefault();
      e.returnValue = '';
      log('beforeunload event: navigation blocked');
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeShouldBlock]);

  // Function to request navigation (from UI)
  const requestNavigation = useCallback((targetPath) => {
    if (!safeShouldBlock) {
      log(`Navigation to "${targetPath}" allowed (no block)`);
      if (typeof navigate === 'function') {
        navigate(targetPath);
      }
      return;
    }
    setPendingPath(targetPath);
    setStatus('syncing');
    log(`Navigation to "${targetPath}" requested, will sync before navigating`);
  }, [safeShouldBlock, navigate]);

  // Effect: when status is 'syncing', run syncFn and then navigate if successful
  useEffect(() => {
    let cancelled = false;
    if (status !== 'syncing' || !pendingPath) return;

    const opId = Symbol('sync-operation');
    asyncOpRef.current = opId;

    const doSync = async () => {
      log(`Starting sync before navigation to "${pendingPath}"`);
      try {
        if (typeof syncFn !== 'function') {
          throw new Error('syncFn is not a function');
        }
        const result = await syncFn();
        if (asyncOpRef.current !== opId || cancelled) {
          log('Sync operation superseded or cancelled');
          return;
        }
        if (result) {
          setPendingPath(null); // Prevent effect re-trigger
          setStatus('synced');
          setSyncError(null);
          log('Sync successful, proceeding with navigation');
          if (unblockRef.current) {
            unblockRef.current();
            unblockRef.current = null;
          }
          if (typeof navigate === 'function') {
            navigate(pendingPath);
          }
        } else {
          setPendingPath(null); // Prevent effect re-trigger
          setStatus('error');
          setSyncError('Sync function returned false');
          log('Sync function returned false, navigation aborted');
        }
      } catch (err) {
        if (asyncOpRef.current !== opId || cancelled) {
          log('Sync operation superseded or cancelled (error case)');
          return;
        }
        setPendingPath(null); // Prevent effect re-trigger
        setStatus('error');
        setSyncError(err);
        log(`Sync error: ${err && err.message ? err.message : err}`);
      } finally {
        if (asyncOpRef.current === opId) {
          setPendingPath(null);
          asyncOpRef.current = null;
        }
      }
    };

    doSync();

    // Cleanup: cancel async op if effect re-runs or unmounts
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, pendingPath, syncFn, navigate]);

  // Reset status to idle if dependencies change and not syncing
  // Removed effect that reset status to 'idle' on dependency change to prevent infinite update loop.

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (unblockRef.current) {
        unblockRef.current();
        unblockRef.current = null;
        log('Navigation blocking cleaned up on unmount');
      }
      asyncOpRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    isBlocking: safeShouldBlock,
    isSyncing: status === 'syncing',
    isSynced: status === 'synced',
    syncError,
    requestNavigation,
    status,
    debugLog: debugLogRef.current
  };
}

export default useSyncBeforeNavigation;
/**
 * Debug configuration for fb_tools extension
 * Controls what debug information gets logged to console
 */

// Debug flag constants (bitwise flags)
export const DEBUG_FLAGS = {
    NONE: 0,
    CONNECTIONS: 1 << 0,      // Log connection attempts (onConnectInput, onConnectOutput)
    CONNECTION_CHANGES: 1 << 1, // Log onConnectionsChange events
    NODE_CREATED: 1 << 2,      // Log when nodes are created
    NODE_EXECUTED: 1 << 3,     // Log when nodes execute
    API_CALLS: 1 << 4,         // Log API requests/responses
    UI_UPDATES: 1 << 5,        // Log UI rendering/updates
    LIBBER: 1 << 6,            // Log libber operations
    SCENE: 1 << 7,             // Log scene operations
    PROMPTS: 1 << 8,           // Log prompt operations
    ALL: 0xFFFFFFFF            // All flags enabled
};

// Current debug configuration
class DebugConfig {
    constructor() {
        // Load from localStorage or default to NONE
        const saved = localStorage.getItem('fb_tools_debug_flags');
        this.flags = saved !== null ? parseInt(saved, 10) : DEBUG_FLAGS.NONE;
    }

    /**
     * Check if a specific debug flag is enabled
     * @param {number} flag - One of the DEBUG_FLAGS constants
     * @returns {boolean}
     */
    isEnabled(flag) {
        return (this.flags & flag) !== 0;
    }

    /**
     * Enable one or more debug flags
     * @param {number} flags - Bitwise OR of DEBUG_FLAGS
     */
    enable(flags) {
        this.flags |= flags;
        this.save();
    }

    /**
     * Disable one or more debug flags
     * @param {number} flags - Bitwise OR of DEBUG_FLAGS
     */
    disable(flags) {
        this.flags &= ~flags;
        this.save();
    }

    /**
     * Set flags to exact value
     * @param {number} flags - Exact flag value
     */
    set(flags) {
        this.flags = flags;
        this.save();
    }

    /**
     * Clear all flags
     */
    clear() {
        this.flags = DEBUG_FLAGS.NONE;
        this.save();
    }

    /**
     * Get current flags value
     * @returns {number}
     */
    get() {
        return this.flags;
    }

    /**
     * Save to localStorage
     */
    save() {
        localStorage.setItem('fb_tools_debug_flags', this.flags.toString());
    }

    /**
     * Get human-readable list of enabled flags
     * @returns {string[]}
     */
    getEnabledFlags() {
        const enabled = [];
        for (const [name, value] of Object.entries(DEBUG_FLAGS)) {
            if (name !== 'NONE' && name !== 'ALL' && this.isEnabled(value)) {
                enabled.push(name);
            }
        }
        return enabled;
    }
}

// Singleton instance
export const debugConfig = new DebugConfig();

/**
 * Conditional logging - only logs if flag is enabled
 * @param {number} flag - DEBUG_FLAGS constant
 * @param {...any} args - Arguments to pass to console.log
 */
export function debugLog(flag, ...args) {
    if (debugConfig.isEnabled(flag)) {
        console.log(...args);
    }
}

/**
 * Console API for easy runtime control
 * Usage in browser console:
 *   window.fbToolsDebug.enable('CONNECTIONS')
 *   window.fbToolsDebug.disable('CONNECTIONS')
 *   window.fbToolsDebug.list()
 */
if (typeof window !== 'undefined') {
    window.fbToolsDebug = {
        enable: (...flagNames) => {
            let flags = 0;
            for (const name of flagNames) {
                if (DEBUG_FLAGS[name.toUpperCase()]) {
                    flags |= DEBUG_FLAGS[name.toUpperCase()];
                } else {
                    console.warn(`Unknown debug flag: ${name}`);
                }
            }
            if (flags) {
                debugConfig.enable(flags);
                console.log(`✓ Enabled: ${flagNames.join(', ')}`);
                console.log(`  Current flags: ${debugConfig.getEnabledFlags().join(', ') || 'NONE'}`);
            }
        },
        disable: (...flagNames) => {
            let flags = 0;
            for (const name of flagNames) {
                if (DEBUG_FLAGS[name.toUpperCase()]) {
                    flags |= DEBUG_FLAGS[name.toUpperCase()];
                } else {
                    console.warn(`Unknown debug flag: ${name}`);
                }
            }
            if (flags) {
                debugConfig.disable(flags);
                console.log(`✓ Disabled: ${flagNames.join(', ')}`);
                console.log(`  Current flags: ${debugConfig.getEnabledFlags().join(', ') || 'NONE'}`);
            }
        },
        clear: () => {
            debugConfig.clear();
            console.log('✓ All debug flags cleared');
        },
        all: () => {
            debugConfig.set(DEBUG_FLAGS.ALL);
            console.log('✓ All debug flags enabled');
        },
        list: () => {
            const enabled = debugConfig.getEnabledFlags();
            console.log('Available flags:', Object.keys(DEBUG_FLAGS).filter(k => k !== 'NONE' && k !== 'ALL'));
            console.log('Currently enabled:', enabled.length > 0 ? enabled : 'NONE');
        },
        get: () => debugConfig.get(),
        FLAGS: DEBUG_FLAGS
    };
    
    console.log('fb_tools debug controls loaded. Type "fbToolsDebug.list()" to see available flags.');
}

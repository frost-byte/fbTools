# Debugging Guide

This guide explains the debug flag system built into fb-tools for troubleshooting and development.

## Overview

The debug system uses **bitwise flags** to provide fine-grained control over logging. By default, all debug logging is **disabled** to keep the console clean. You can enable specific categories of debugging at runtime through the browser console without modifying any code.

## Quick Start

1. Open ComfyUI in your browser
2. Open the browser developer console (F12)
3. Enable debugging for specific features:

```javascript
// Enable connection debugging
fbToolsDebug.enable('CONNECTIONS')

// Enable multiple categories at once
fbToolsDebug.enable('CONNECTIONS', 'API_CALLS', 'UI_UPDATES')

// Enable all debugging
fbToolsDebug.all()
```

## Available Debug Flags

| Flag | Description |
|------|-------------|
| `NONE` | No debug output (default) |
| `CONNECTIONS` | Log connection attempts between nodes |
| `CONNECTION_CHANGES` | Log connection state changes (connect/disconnect) |
| `NODE_CREATED` | Log node creation events |
| `NODE_EXECUTED` | Log node execution and results |
| `API_CALLS` | Log API requests and responses |
| `UI_UPDATES` | Log UI update operations |
| `LIBBER` | Log Libber-specific operations (templates, state) |
| `SCENE` | Log Scene-specific operations |
| `PROMPTS` | Log Prompt CRUD operations |
| `ALL` | Enable all debug categories |

## Console API Reference

The debug system is accessible via `window.fbToolsDebug` in the browser console.

### Enable Flags

```javascript
// Enable one or more flags
fbToolsDebug.enable('CONNECTIONS')
fbToolsDebug.enable('CONNECTIONS', 'API_CALLS', 'SCENE')

// Enable all debugging
fbToolsDebug.all()
```

### Disable Flags

```javascript
// Disable specific flags
fbToolsDebug.disable('CONNECTIONS')
fbToolsDebug.disable('CONNECTIONS', 'API_CALLS')

// Disable all debugging (silent mode)
fbToolsDebug.clear()
```

### Query Debug State

```javascript
// List available flags and show which are enabled
fbToolsDebug.list()

// Get current numeric flag value
fbToolsDebug.get()
```

## Persistence

Debug settings are **automatically saved** to browser localStorage and will persist across:
- Page refreshes
- Browser restarts
- ComfyUI server restarts

To reset to default (all flags off):
```javascript
fbToolsDebug.clear()
```

## How It Works

### Bitwise Flag System

The debug system uses bitwise operations for efficient flag management:

```javascript
// Each flag is a power of 2
CONNECTIONS       = 1 << 0  // 1
CONNECTION_CHANGES = 1 << 1  // 2
NODE_CREATED      = 1 << 2  // 4
API_CALLS         = 1 << 3  // 8
// ... and so on

// Multiple flags are combined with bitwise OR
CONNECTIONS | API_CALLS = 9  // 0b00001001
```

This allows:
- Efficient storage (single number)
- Fast checking (bitwise AND)
- Easy combination of multiple flags

### In Code

Developers can use the debug system in JavaScript modules:

```javascript
import { debugLog, DEBUG_FLAGS } from './utils/debug_config.js';

// Log only when CONNECTIONS flag is enabled
debugLog(DEBUG_FLAGS.CONNECTIONS, 'Connection attempt:', data);

// Log with multiple conditions
debugLog(DEBUG_FLAGS.API_CALLS | DEBUG_FLAGS.SCENE, 'Scene API call:', endpoint);
```

## Common Debugging Scenarios

### Troubleshooting Node Connections

```javascript
// Enable connection debugging
fbToolsDebug.enable('CONNECTIONS', 'CONNECTION_CHANGES')

// Now try connecting nodes in ComfyUI
// Console will show detailed connection information
```

### Debugging API Issues

```javascript
// Enable API and relevant feature debugging
fbToolsDebug.enable('API_CALLS', 'SCENE', 'PROMPTS')

// Execute nodes or trigger operations
// Console will show API requests/responses
```

### Debugging UI Updates

```javascript
// Enable UI debugging
fbToolsDebug.enable('UI_UPDATES', 'LIBBER')

// Interact with UI elements
// Console will show update operations
```

### Full System Debugging

```javascript
// Enable everything (verbose!)
fbToolsDebug.all()

// When done, disable to reduce noise
fbToolsDebug.clear()
```

## Development Guidelines

### Adding New Debug Points

1. Import the debug utilities:
```javascript
import { debugLog, DEBUG_FLAGS } from './utils/debug_config.js';
```

2. Use appropriate flag for your log:
```javascript
debugLog(DEBUG_FLAGS.API_CALLS, 'Making request to:', endpoint);
```

3. Choose the most specific flag:
   - Connection-related: `CONNECTIONS` or `CONNECTION_CHANGES`
   - API requests: `API_CALLS`
   - Feature-specific: `LIBBER`, `SCENE`, `PROMPTS`
   - Generic UI: `UI_UPDATES`

### Adding New Flags

To add a new debug category:

1. Edit `js/utils/debug_config.js`
2. Add new flag constant:
```javascript
export const DEBUG_FLAGS = {
    // ... existing flags ...
    MY_NEW_FEATURE: 1 << 10,  // Use next available bit
    ALL: (1 << 11) - 1        // Update ALL to include new bit
};
```

3. Update flag mappings:
```javascript
const FLAG_NAMES = {
    // ... existing mappings ...
    'MY_NEW_FEATURE': DEBUG_FLAGS.MY_NEW_FEATURE,
};

const FLAG_DESCRIPTIONS = {
    // ... existing descriptions ...
    MY_NEW_FEATURE: 'My feature description',
};
```

## Technical Details

### Storage Location

Debug settings are stored in browser localStorage:
- **Key**: `fb_tools_debug_flags`
- **Value**: Numeric bitwise flag value
- **Scope**: Per browser, per domain

### Default State

By default, all flags are **disabled** (`DEBUG_FLAGS.NONE = 0`). This ensures:
- Clean console output in production
- No performance impact from logging
- Opt-in debugging when needed

### Performance

The debug system is designed for minimal overhead:
- Flag checks are simple bitwise AND operations
- No string comparisons or complex logic
- Logs only execute when flags are enabled
- No impact when debugging is disabled

## Troubleshooting

### Debug API Not Available

If `fbToolsDebug` is undefined:
1. Check browser console for errors loading `debug_config.js`
2. Verify the file exists: `custom_nodes/comfyui-fbTools/js/utils/debug_config.js`
3. Clear browser cache and reload
4. Check ComfyUI console for extension loading errors

### Flags Not Persisting

If debug flags reset on page reload:
1. Check browser localStorage is enabled
2. Verify no browser extensions are blocking storage
3. Check for localStorage quota errors in console
4. Try: `localStorage.getItem('fb_tools_debug_flags')`

### Too Much Output

If console is flooded with logs:
```javascript
// Disable specific noisy flags
fbToolsDebug.disable('CONNECTION_CHANGES')

// Or disable all
fbToolsDebug.clear()
```

### Missing Expected Logs

If you enabled a flag but see no logs:
1. Verify flag is enabled: `fbToolsDebug.list()`
2. Check numeric value: `fbToolsDebug.get()`
3. Ensure feature being debugged is actually executing
4. Try enabling `ALL` to see if any logs appear

## Best Practices

1. **Start Narrow**: Enable only the flags you need
2. **Disable When Done**: Clear flags after debugging to reduce noise
3. **Use Combinations**: Enable related flags together for context
4. **Document**: When reporting issues, mention which flags were enabled
5. **Check First**: Run `fbToolsDebug.list()` before assuming flags are off

## Examples

### Example: Debugging Connection Issues

```javascript
// 1. Enable connection debugging
fbToolsDebug.enable('CONNECTIONS', 'CONNECTION_CHANGES')

// 2. Try connecting SceneSelect to ScenePromptManager
// Console output:
// 🔌 SceneSelect.onConnectOutput called:
//   outputIndex: 0
//   inputType: SCENE_INFO
//   ...
// 🔗 ScenePromptManager.onConnectionsChange called:
//   type: 1 (input)
//   connected: true
//   ...

// 3. Disable when done
fbToolsDebug.disable('CONNECTIONS', 'CONNECTION_CHANGES')
```

### Example: Debugging Prompt Processing

```javascript
// 1. Enable prompt and API debugging
fbToolsDebug.enable('PROMPTS', 'API_CALLS')

// 2. Click "Process" button in ScenePromptManager
// Console output:
// [API] POST /fbtools/scene/process_compositions
// [PROMPTS] Processing 3 compositions
// [PROMPTS] Result: { prompt_dict: {...} }

// 3. Clear all flags
fbToolsDebug.clear()
```

### Example: Development Session

```javascript
// At start of debugging session
fbToolsDebug.enable('CONNECTIONS', 'API_CALLS', 'UI_UPDATES')

// Work on features...

// When switching focus
fbToolsDebug.disable('CONNECTIONS')
fbToolsDebug.enable('SCENE', 'PROMPTS')

// At end of session
fbToolsDebug.clear()
```

## Related Documentation

- [Testing Guide](TESTING_SCENE_TABS.md) - UI testing procedures
- [Scene Prompt System](SCENE_PROMPT_SYSTEM.md) - Scene management architecture
- [Libber Nodes](LIBBER_NODES_README.md) - Template system documentation

## Support

If you encounter issues with the debug system:
1. Check this documentation for solutions
2. Verify browser console for errors
3. Test with `fbToolsDebug.all()` to see if any logging works
4. Report issues with browser/OS details and enabled flags

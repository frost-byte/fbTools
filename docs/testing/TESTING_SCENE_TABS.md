# Testing the Three-Tab ScenePromptManager

## Setup
1. Refresh your browser (no ComfyUI service restart needed!)
2. Open ComfyUI in browser
3. Add a ScenePromptManager node to your workflow

## Test Plan

### Phase 1: Define Tab (Baseline Functionality)

**Test 1.1: Add a raw prompt**
1. Go to Define tab (should be default)
2. In the new prompt row at bottom:
   - Key: `character`
   - Value: `beautiful woman with long hair`
   - Type: `raw`
   - Category: `character`
3. Click ➕ button
4. Verify: Toast shows "Added 'character'"
5. Verify: Prompt appears in table

**Test 1.2: Add a libber prompt**
1. In new prompt row:
   - Key: `setting`
   - Value: `in a %location%, %time_of_day%`
   - Type: Select `libber`
2. Verify: Libber dropdown becomes enabled
3. Verify: Libber dropdown is populated (may show "none" if no libbers exist)
4. Click ➕ button
5. Verify: Prompt appears with libber type

**Test 1.3: Remove a prompt**
1. Click ➖ button on any prompt row
2. Verify: Toast shows "Removed 'key_name'"
3. Verify: Prompt disappears from table

**Test 1.4: Apply changes**
1. Edit an existing prompt's value
2. Click "✓ Apply Changes" button
3. Verify: Toast shows "Applied N prompts"
4. Check collection_json widget - should contain updated prompts

### Phase 2: Compose Tab (New Functionality)

**Test 2.1: Navigate to Compose tab**
1. Click "🎨 Compose" tab button
2. Verify: Tab button has active styling (border-bottom, bold)
3. Verify: Content shows composition table with help text

**Test 2.2: Add a composition**
1. In new composition row:
   - Name: `final_prompt`
   - Select multiple keys using Ctrl/Cmd+click in multi-select
2. Click ➕ button
3. Verify: Toast shows "Added composition 'final_prompt'"
4. Verify: Composition appears in table with selected keys as tags

**Test 2.3: Add key to existing composition**
1. Click "+ Add Key" button on a composition row
2. Enter a prompt key when prompted
3. Verify: Key is added to the composition's tag list

**Test 2.4: Remove key from composition**
1. Click the × button on a key tag
2. Verify: Toast shows "Removed key 'key_name'"
3. Verify: Key tag disappears

**Test 2.5: Remove entire composition**
1. Click 🗑️ button on a composition row
2. Verify: Toast shows "Removed composition 'name'"
3. Verify: Composition disappears from table

**Test 2.6: Apply compositions**
1. Edit composition name if needed
2. Click "✓ Apply Compositions" button
3. Verify: Toast shows "Applied N compositions"
4. Check collection_json widget - should contain compositions object

### Phase 3: View Tab (Preview Functionality)

**Test 3.1: Navigate to View tab**
1. Click "👁️ View" tab button
2. Verify: Tab has active styling
3. If no compositions: Shows "No compositions yet. Create them in the Compose tab."
4. If compositions exist: Shows dropdown and textarea

**Test 3.2: View composition output**
1. Ensure you have at least one composition created
2. Execute the node (Queue Prompt in ComfyUI)
3. Wait for execution to complete
4. Go to View tab
5. Verify: Dropdown shows all composition names
6. Verify: Textarea shows processed output for first composition
7. Verify: Character count is displayed

**Test 3.3: Switch between compositions**
1. Select different composition from dropdown
2. Verify: Textarea updates to show that composition's output
3. Verify: Character count updates correctly

**Test 3.4: Copy to clipboard**
1. Click "📋 Copy" button
2. Verify: Toast shows "Copied to clipboard"
3. Paste somewhere to verify contents

### Phase 4: Integration Tests

**Test 4.1: Tab switching preserves data**
1. Add prompts in Define tab
2. Switch to Compose tab
3. Switch back to Define tab
4. Verify: All prompts still there

**Test 4.2: Backend round-trip**
1. Add several prompts in Define tab
2. Click Apply Changes
3. Add compositions in Compose tab
4. Click Apply Compositions
5. Execute the node (Queue Prompt)
6. Check all tabs after execution:
   - Define: All prompts should be there
   - Compose: All compositions should be there
   - View: Should show composed outputs

**Test 4.3: Libber substitution in compositions**
1. Create a libber prompt with variables (e.g., `%location%`)
2. Create a composition including that prompt
3. Execute the node
4. Go to View tab
5. Verify: Variables are substituted in output (if libber exists)

**Test 4.4: Persistence test**
1. Create prompts and compositions
2. Save the workflow
3. Reload the browser
4. Load the workflow
5. Verify: All prompts and compositions are restored

### Phase 5: Edge Cases

**Test 5.1: Empty state**
1. Fresh node with no data
2. Verify: Define tab shows empty table with new row
3. Verify: Compose tab shows "0 compositions"
4. Verify: View tab shows "No compositions yet" message

**Test 5.2: Duplicate key/name validation**
1. Try to add prompt with existing key
2. Verify: Toast shows warning "Key 'X' already exists"
3. Try to add composition with existing name
4. Verify: Toast shows warning "Composition 'X' already exists"

**Test 5.3: Missing required fields**
1. Try to add prompt without key
2. Verify: Toast shows "Key required"
3. Try to add composition without name
4. Verify: Toast shows "Composition name required"
5. Try to add composition without selecting keys
6. Verify: Toast shows "Select at least one prompt key"

**Test 5.4: Libber dropdown behavior**
1. Create prompt with type=raw, libber dropdown should be disabled
2. Change type to libber
3. Verify: Dropdown enables and may auto-populate with libbers
4. Change type back to raw
5. Verify: Dropdown disables and resets to "none"

## Success Criteria

✅ All tabs render correctly
✅ Tab switching works smoothly
✅ Define tab: Add/edit/remove prompts works
✅ Compose tab: Add/edit/remove compositions works
✅ View tab: Shows composed outputs correctly
✅ Apply buttons update collection_json widget
✅ Backend execution updates all tabs
✅ Data persists in prompts.json
✅ No console errors
✅ Toast notifications work correctly

## Debugging Tips

### If tabs don't appear:
- Check browser console for JavaScript errors
- Verify [scene.js](js/nodes/scene.js) was updated
- Hard refresh browser (Ctrl+Shift+R)

### If backend data doesn't update:
- Check extension.py has prompt_dict output
- Check prompt_models.py has compositions field
- Verify ComfyUI service is running 0.3.77

### If libber dropdown is empty:
- Check if any libbers exist in the system
- Use LibberManager to create a libber first
- Check browser console for API errors

### Browser console commands:
```javascript
// Check current state
console.log("Active Tab:", activeTab);
console.log("Prompts:", currentPromptsData);
console.log("Compositions:", currentCompositionsData);
console.log("Prompt Dict:", currentPromptDict);
```

## Expected File Changes

**Modified:**
- [js/nodes/scene.js](js/nodes/scene.js) - Three-tab UI implementation

**Created:**
- [SCENE_PROMPT_MANAGER_TABS.md](SCENE_PROMPT_MANAGER_TABS.md) - Feature documentation
- This file: Testing guide

**Backend (already completed):**
- [src/fb_tools/prompt_models.py](src/fb_tools/prompt_models.py) - compositions field
- [extension.py](extension.py) - prompt_dict output

## Report Issues

If you find bugs or unexpected behavior:
1. Check browser console for errors
2. Note which test case failed
3. Note what you expected vs what happened
4. Check collection_json widget contents
5. Report with all above information

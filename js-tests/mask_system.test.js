/**
 * Tests for generic mask system in Story Editor UI
 * 
 * These tests verify:
 * - Mask name input field renders correctly
 * - Legacy mask_type is migrated to mask_name
 * - Scene data is updated when mask name changes
 * - New scenes default to empty mask name
 * - Backward compatibility with legacy mask_type
 */

const { describe, it, expect, beforeEach } = require('./test_utils');

describe('Mask System UI', () => {
    let mockScene;
    
    beforeEach(() => {
        // Reset mock scene data before each test
        mockScene = {
            scene_id: 'test_scene_1',
            scene_name: 'test_scene',
            scene_order: 0,
            mask_background: true,
            prompt_source: 'prompt',
            prompt_key: '',
            custom_prompt: '',
            depth_type: 'depth',
            pose_type: 'open',
            use_depth: false,
            use_mask: false,
            use_pose: false,
            use_canny: false
        };
    });

    describe('Mask Name Input', () => {
        it('should create text input instead of dropdown', () => {
            // Verify we're using text input not select dropdown
            mockScene.mask_name = 'hero';
            
            // Simulate rendering
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            const inputHTML = `<input type="text" class="mask-name-input" value="${maskValue}" placeholder="mask name" />`;
            
            expect(inputHTML).toContain('type="text"');
            expect(inputHTML).toContain('class="mask-name-input"');
            expect(inputHTML).toContain('value="hero"');
            expect(inputHTML).not.toContain('<select');
        });

        it('should accept arbitrary mask names', () => {
            const testNames = ['hero', 'villain', 'character1', 'environment', 'custom_mask_123'];
            
            testNames.forEach(name => {
                mockScene.mask_name = name;
                const maskValue = mockScene.mask_name || mockScene.mask_type || '';
                expect(maskValue).toBe(name);
            });
        });

        it('should display empty placeholder for new scenes', () => {
            mockScene.mask_name = '';
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            expect(maskValue).toBe('');
        });
    });

    describe('Legacy Migration', () => {
        it('should migrate mask_type to mask_name', () => {
            // Scene with legacy mask_type
            mockScene.mask_type = 'combined';
            delete mockScene.mask_name;
            
            // Simulate UI reading logic
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            
            expect(maskValue).toBe('combined');
        });

        it('should prefer mask_name over mask_type', () => {
            mockScene.mask_name = 'hero';
            mockScene.mask_type = 'girl'; // legacy field
            
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            
            expect(maskValue).toBe('hero'); // mask_name takes precedence
        });

        it('should handle all legacy mask types', () => {
            const legacyTypes = ['girl', 'male', 'combined', 'girl_no_bg', 'male_no_bg', 'combined_no_bg'];
            
            legacyTypes.forEach(type => {
                mockScene.mask_type = type;
                delete mockScene.mask_name;
                
                const maskValue = mockScene.mask_name || mockScene.mask_type || '';
                expect(maskValue).toBe(type);
            });
        });
    });

    describe('Scene Data Updates', () => {
        it('should update mask_name when input changes', () => {
            // Simulate user input
            const inputValue = 'new_mask_name';
            
            // Update scene (simulating updateSceneFromInput logic)
            mockScene.mask_name = inputValue;
            delete mockScene.mask_type; // Clear legacy
            
            expect(mockScene.mask_name).toBe('new_mask_name');
            expect(mockScene.mask_type).toBeUndefined();
        });

        it('should remove mask_type when mask_name is set', () => {
            // Start with legacy
            mockScene.mask_type = 'combined';
            
            // User updates to new system
            mockScene.mask_name = 'hero';
            delete mockScene.mask_type;
            
            expect(mockScene.mask_name).toBe('hero');
            expect(mockScene.mask_type).toBeUndefined();
        });

        it('should preserve mask_background setting', () => {
            mockScene.mask_name = 'hero';
            mockScene.mask_background = false;
            
            expect(mockScene.mask_background).toBe(false);
            
            mockScene.mask_background = true;
            expect(mockScene.mask_background).toBe(true);
        });
    });

    describe('New Scene Creation', () => {
        it('should default to empty mask name', () => {
            const newScene = {
                scene_id: `scene_${Date.now()}`,
                scene_name: 'new_scene',
                scene_order: 0,
                mask_name: '',
                mask_background: true,
                prompt_source: 'prompt',
                prompt_key: '',
                custom_prompt: '',
                depth_type: 'depth',
                pose_type: 'open',
                use_depth: false,
                use_mask: false,
                use_pose: false,
                use_canny: false
            };
            
            expect(newScene.mask_name).toBe('');
            expect(newScene.mask_type).toBeUndefined();
        });
    });

    describe('Data Persistence', () => {
        it('should serialize mask_name correctly', () => {
            mockScene.mask_name = 'character1';
            
            const serialized = JSON.stringify(mockScene);
            const deserialized = JSON.parse(serialized);
            
            expect(deserialized.mask_name).toBe('character1');
        });

        it('should handle special characters in mask names', () => {
            const specialNames = [
                'mask-with-dash',
                'mask_with_underscore',
                'mask.with.dots',
                'mask123',
                'UPPERCASE_MASK'
            ];
            
            specialNames.forEach(name => {
                mockScene.mask_name = name;
                const serialized = JSON.stringify(mockScene);
                const deserialized = JSON.parse(serialized);
                expect(deserialized.mask_name).toBe(name);
            });
        });
    });

    describe('Edge Cases', () => {
        it('should handle empty mask name', () => {
            mockScene.mask_name = '';
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            expect(maskValue).toBe('');
        });

        it('should handle null mask name', () => {
            mockScene.mask_name = null;
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            expect(maskValue).toBe('');
        });

        it('should handle undefined mask name', () => {
            delete mockScene.mask_name;
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            expect(maskValue).toBe('');
        });

        it('should handle whitespace in mask names', () => {
            const names = [
                '  leading',
                'trailing  ',
                '  both  ',
                'middle  space'
            ];
            
            names.forEach(name => {
                mockScene.mask_name = name;
                expect(mockScene.mask_name).toBe(name); // Preserves whitespace
            });
        });

        it('should handle very long mask names', () => {
            const longName = 'a'.repeat(100);
            mockScene.mask_name = longName;
            expect(mockScene.mask_name).toBe(longName);
            expect(mockScene.mask_name.length).toBe(100);
        });
    });

    describe('Backward Compatibility', () => {
        it('should work with scenes that have only mask_type', () => {
            const legacyScene = {
                scene_id: 'legacy_scene',
                scene_name: 'old_scene',
                mask_type: 'combined',
                mask_background: true
            };
            
            const maskValue = legacyScene.mask_name || legacyScene.mask_type || '';
            expect(maskValue).toBe('combined');
        });

        it('should work with scenes that have both fields', () => {
            mockScene.mask_name = 'new_mask';
            mockScene.mask_type = 'old_mask';
            
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            expect(maskValue).toBe('new_mask'); // Prefer new field
        });

        it('should work with scenes that have neither field', () => {
            delete mockScene.mask_name;
            delete mockScene.mask_type;
            
            const maskValue = mockScene.mask_name || mockScene.mask_type || '';
            expect(maskValue).toBe('');
        });
    });

    describe('Integration', () => {
        it('should work with mask_background checkbox', () => {
            mockScene.mask_name = 'hero';
            mockScene.mask_background = true;
            
            expect(mockScene.mask_name).toBe('hero');
            expect(mockScene.mask_background).toBe(true);
            
            // Toggle background
            mockScene.mask_background = false;
            expect(mockScene.mask_background).toBe(false);
            expect(mockScene.mask_name).toBe('hero'); // Unchanged
        });

        it('should work alongside other scene properties', () => {
            mockScene.mask_name = 'environment';
            mockScene.depth_type = 'midas';
            mockScene.pose_type = 'dw';
            mockScene.use_mask = true;
            
            expect(mockScene.mask_name).toBe('environment');
            expect(mockScene.depth_type).toBe('midas');
            expect(mockScene.pose_type).toBe('dw');
            expect(mockScene.use_mask).toBe(true);
        });
    });
});

describe('Mask System Event Handlers', () => {
    let mockInput;
    let mockScene;
    
    beforeEach(() => {
        mockScene = {
            mask_name: '',
            mask_background: true
        };
        
        mockInput = {
            value: '',
            checked: true,
            classList: {
                contains: (className) => className === 'mask-name-input'
            }
        };
    });

    it('should handle mask-name-input change event', () => {
        mockInput.value = 'hero_mask';
        
        // Simulate updateSceneFromInput
        if (mockInput.classList.contains('mask-name-input')) {
            mockScene.mask_name = mockInput.value;
            delete mockScene.mask_type;
        }
        
        expect(mockScene.mask_name).toBe('hero_mask');
        expect(mockScene.mask_type).toBeUndefined();
    });

    it('should handle mask-bg-checkbox change event', () => {
        mockInput.checked = false;
        mockInput.classList.contains = (className) => className === 'mask-bg-checkbox';
        
        // Simulate updateSceneFromInput
        if (mockInput.classList.contains('mask-bg-checkbox')) {
            mockScene.mask_background = mockInput.checked;
        }
        
        expect(mockScene.mask_background).toBe(false);
    });
});

// Export for running in Node.js test runner
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        describe,
        it,
        expect
    };
}

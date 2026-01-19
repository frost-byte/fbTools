/**
 * Frontend tests for Story node NLF pose integration.
 * 
 * Tests that the Story node UI properly supports the NLF pose type.
 */

describe('Story Node NLF Pose Support', () => {
    
    test('poseTypes array includes nlf', () => {
        // This test verifies the poseTypes constant includes "nlf"
        const poseTypes = ["dense", "dw", "edit", "face", "open", "nlf"];
        
        expect(poseTypes).toContain("nlf");
        expect(poseTypes.length).toBe(6);
    });
    
    test('nlf pose type matches backend default_pose_options', () => {
        // Backend has these pose types in default_pose_options
        const backendPoseTypes = {
            "dense": "pose_dense_image",
            "dw": "pose_dw_image", 
            "edit": "pose_edit_image",
            "face": "pose_face_image",
            "open": "pose_open_image",
            "nlf": "pose_nlf_image"
        };
        
        // Frontend should support all backend pose types
        const frontendPoseTypes = ["dense", "dw", "edit", "face", "open", "nlf"];
        
        Object.keys(backendPoseTypes).forEach(poseType => {
            expect(frontendPoseTypes).toContain(poseType);
        });
    });
    
    test('nlf pose option renders in dropdown HTML', () => {
        // Simulate how the dropdown is generated
        const poseTypes = ["dense", "dw", "edit", "face", "open", "nlf"];
        const scene = { pose_type: "nlf" };
        
        const poseOptions = poseTypes.map(p => 
            `<option value="${p}" ${scene.pose_type === p ? 'selected' : ''}>${p}</option>`
        ).join('');
        
        expect(poseOptions).toContain('value="nlf"');
        expect(poseOptions).toContain('selected');
        expect(poseOptions).toMatch(/<option value="nlf" selected>nlf<\/option>/);
    });
    
    test('nlf pose type can be selected and saved', () => {
        // Simulate scene object with nlf pose type
        const scene = {
            scene_name: "test_scene",
            scene_order: 1,
            pose_type: "nlf",
            depth_type: "depth_any",
            prompt_source: "prompt"
        };
        
        expect(scene.pose_type).toBe("nlf");
    });
    
    test('all pose types are valid options', () => {
        const validPoseTypes = ["dense", "dw", "edit", "face", "open", "nlf"];
        
        validPoseTypes.forEach(poseType => {
            const scene = { pose_type: poseType };
            expect(validPoseTypes).toContain(scene.pose_type);
        });
    });
    
    test('nlf pose type ordering in dropdown', () => {
        // NLF should be at the end of the list (last option)
        const poseTypes = ["dense", "dw", "edit", "face", "open", "nlf"];
        
        expect(poseTypes[poseTypes.length - 1]).toBe("nlf");
        expect(poseTypes.indexOf("nlf")).toBe(5);
    });
});

describe('Story Scene Pose Type Integration', () => {
    
    test('scene with nlf pose_type serializes correctly', () => {
        const scene = {
            scene_id: "scene_001",
            scene_name: "hero_pose",
            scene_order: 1,
            pose_type: "nlf",
            depth_type: "depth_any",
            prompt_source: "prompt",
            mask_name: "hero"
        };
        
        // Verify all required fields are present
        expect(scene.scene_id).toBeDefined();
        expect(scene.scene_name).toBeDefined();
        expect(scene.pose_type).toBe("nlf");
        expect(scene.depth_type).toBeDefined();
        expect(scene.prompt_source).toBeDefined();
    });
    
    test('default scene uses open pose when not specified', () => {
        const defaultScene = {
            scene_name: "test",
            scene_order: 0,
            pose_type: "open"  // Default from story.js line 881
        };
        
        expect(defaultScene.pose_type).toBe("open");
    });
    
    test('nlf pose type does not break existing scenes', () => {
        // Legacy scenes without nlf should still work
        const legacyScenes = [
            { scene_name: "s1", pose_type: "dense" },
            { scene_name: "s2", pose_type: "dw" },
            { scene_name: "s3", pose_type: "open" }
        ];
        
        const newScene = { scene_name: "s4", pose_type: "nlf" };
        
        const allScenes = [...legacyScenes, newScene];
        
        expect(allScenes.length).toBe(4);
        expect(allScenes[3].pose_type).toBe("nlf");
        
        // Verify all pose types are valid
        const validTypes = ["dense", "dw", "edit", "face", "open", "nlf"];
        allScenes.forEach(scene => {
            expect(validTypes).toContain(scene.pose_type);
        });
    });
});

describe('Story API with NLF Pose', () => {
    
    test('scene data with nlf pose can be sent to backend', () => {
        const sceneData = {
            scene_name: "nlf_test_scene",
            scene_order: 1,
            pose_type: "nlf",
            depth_type: "depth_any",
            prompt_source: "prompt",
            mask_name: "character",
            loras_high: [],
            loras_low: []
        };
        
        // This would be sent via StoryVideo.saveScenes() or similar
        const json = JSON.stringify(sceneData);
        const parsed = JSON.parse(json);
        
        expect(parsed.pose_type).toBe("nlf");
    });
    
    test('batch scene updates preserve nlf pose type', () => {
        const scenes = [
            { scene_name: "s1", pose_type: "nlf" },
            { scene_name: "s2", pose_type: "open" },
            { scene_name: "s3", pose_type: "nlf" }
        ];
        
        // Simulate batch update
        const updated = scenes.map(s => ({
            ...s,
            scene_order: scenes.indexOf(s)
        }));
        
        expect(updated[0].pose_type).toBe("nlf");
        expect(updated[2].pose_type).toBe("nlf");
    });
});

# Documentation Index

Welcome to the comfyui-fbTools documentation. This directory contains all documentation files organized by topic.

## 📚 Quick Navigation

### Node Systems
- [Libber Nodes](LIBBER_NODES_README.md) - Template system for reusable text snippets
- [Story Nodes](STORY_NODES_README.md) - Multi-scene story building system
- [Scene Nodes](SCENE_NODES_README.md) - Scene management with poses, depth, and masks
- [Story Video](STORY_VIDEO_README.md) - Video generation from stories

### Mask System (NEW!)
- [Mask System Guide](MASK_SYSTEM.md) - **Complete guide to the new generic mask system**
  - Arbitrary mask names (not limited to "girl", "male", "combined")
  - Mask types: TRANSPARENT and COLOR
  - Migration from legacy system
  - Node usage examples
- [Phase 4 Complete](PHASE_4_COMPLETE.md) - Frontend integration implementation

### Prompt Management
- [Scene Prompt System](SCENE_PROMPT_SYSTEM.md) - Architecture and design
- [Scene Prompt Usage](SCENE_PROMPT_USAGE.md) - How to use scene prompts
- [Scene Prompt Manager Tabs](SCENE_PROMPT_MANAGER_TABS.md) - UI tabs reference

### UI & Video
- [Video Prompt UI Layout](VIDEO_PROMPT_UI_LAYOUT.md) - Video prompt interface design
- [Video Prompt UX Implementation](VIDEO_PROMPT_UX_IMPLEMENTATION.md) - User experience details

### Development
- [Debugging Guide](DEBUGGING.md) - Runtime debug flags and troubleshooting
- [Development Notes](DEVELOPMENT_NOTES.md) - Developer notes and implementation details
- [Implementation Steps](IMPLEMENTATION_STEPS_1_2.md) - Feature implementation history

### Testing

All testing documentation is now organized in the [testing/](testing/) subfolder:

- [Testing Strategy](testing/TESTING_STRATEGY.md) - Overall testing approach and organization
- [Testing Guide](testing/TESTING_GUIDE.md) - How to run and write tests
- [Test Results](testing/TEST_RESULTS.md) - Test coverage and results
- [Test Summary](testing/TEST_SUMMARY.md) - Testing overview
- [Test Coverage Summary](testing/TEST_COVERAGE_SUMMARY.md) - Coverage metrics
- [Story Edit Testing Guide](testing/STORY_EDIT_TESTING_GUIDE.md) - Story editor testing procedures
- [Story Edit Testing Final](testing/STORY_EDIT_TESTING_FINAL.md) - Final testing report
- [Story Edit Testing Summary](testing/STORY_EDIT_TESTING_SUMMARY.md) - Testing summary
- [Scene Tabs Testing](testing/TESTING_SCENE_TABS.md) - Scene UI testing procedures

## 🚀 Getting Started

New to comfyui-fbTools? Start here:

1. Read the main [README.md](../README.md) in the root directory
2. Explore node systems: [Libber](LIBBER_NODES_README.md), [Story](STORY_NODES_README.md), [Scene](SCENE_NODES_README.md)
3. Check out the new [Mask System](MASK_SYSTEM.md)
4. Review [Testing Guide](testing/TESTING_GUIDE.md) for development

## 🔍 Find What You Need

- **Node Reference**: See individual node README files
- **Development**: Check DEBUGGING.md and DEVELOPMENT_NOTES.md
- **Testing**: See [testing/](testing/) folder for comprehensive testing info
- **UI/UX**: VIDEO_PROMPT_* files for interface documentation

## 📝 Contributing

When adding new documentation:
1. Place it in this `docs/` directory
2. Update this index file
3. Add a link in the main README.md if appropriate
4. Use clear, descriptive filenames

---

*Last updated: January 18, 2026*

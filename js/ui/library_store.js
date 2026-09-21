/**
 * Shared library state for the panel's asset editors.
 *
 * Subjects, backgrounds, outfits, camera/sound presets, plus the media file
 * lists, LLM vision flags and SAM2 status the asset editors need. Held in one
 * module-level object so the Compose editor and the Assets tab always see the
 * same data. After any save/delete an editor calls notifyLibraryChanged(kind)
 * and every interested view refreshes itself from `lib`.
 */

export const lib = {
    subjects:       [],
    backgrounds:    [],
    cameraPresets:  [],
    soundPresets:   [],
    outfits:        {},    // id -> {name, description, tags, reference_images}
    sam2:           null,  // null = unchecked; {available, packages_ok, model_file, ...}
    mediaInImages:  [],
    mediaOutImages: [],
    mediaInVideos:  [],
    mediaOutVideos: [],
    llmLoaded:      null,
    llmVision:      false,
    llmNativeVideo: false,
    // Set by the Compose editor so history entries can name the open composition.
    getCompositionName: () => "",
};

export const LIBRARY_CHANGED = "fbt:library-changed";

/**
 * Tell every view that a library kind changed
 * (kind: "subjects" | "backgrounds" | "outfits" | "cameraPresets" | "soundPresets").
 * `extra` carries kind-specific details, e.g. {deletedId} for a removed background.
 */
export function notifyLibraryChanged(kind, extra = {}) {
    document.dispatchEvent(new CustomEvent(LIBRARY_CHANGED, { detail: { kind, ...extra } }));
}

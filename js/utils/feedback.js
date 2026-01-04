/**
 * UI Feedback utilities for displaying success/error overlays
 */

/**
 * Show a full-screen overlay with a message
 * @param {Object} options - Overlay options
 * @param {HTMLElement} options.container - The container element to append the overlay to
 * @param {string} options.message - Main message to display (default: "Success")
 * @param {string} options.details - Optional details text
 * @param {string} options.type - Type of message: "success" or "error" (default: "success")
 * @param {number} options.duration - Duration in ms before auto-hide (default: 2000)
 */
export function showOverlay(options) {
    const {
        container,
        message = "Success",
        details = "",
        type = "success",
        duration = 2000
    } = options;
    
    if (!container) {
        console.warn("showOverlay: container is required");
        return;
    }
    
    // Determine colors based on type
    const borderColor = type === "success" ? "#4CAF50" : "#f44336";
    const textColor = type === "success" ? "#4CAF50" : "#f44336";
    const icon = type === "success" ? "✓" : "✗";
    
    // Check if overlay already exists
    let overlay = container.querySelector('.feedback-overlay');
    let messageEl, detailsEl;
    
    if (!overlay) {
        // Create overlay structure
        overlay = document.createElement('div');
        overlay.className = 'feedback-overlay';
        overlay.style.cssText = `
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            z-index: 10000;
            align-items: center;
            justify-content: center;
        `;
        
        const contentBox = document.createElement('div');
        contentBox.style.cssText = `
            padding: 30px;
            text-align: center;
            color: var(--fg-color);
            background: var(--comfy-menu-bg);
            border-radius: 8px;
            border: 2px solid ${borderColor};
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        `;
        
        messageEl = document.createElement('p');
        messageEl.className = 'feedback-message';
        messageEl.style.cssText = `
            margin: 10px 0;
            color: ${textColor};
            font-weight: 600;
            font-size: 16px;
        `;
        
        detailsEl = document.createElement('p');
        detailsEl.className = 'feedback-details';
        detailsEl.style.cssText = `
            margin: 10px 0;
            opacity: 0.7;
            font-size: 12px;
        `;
        
        contentBox.appendChild(messageEl);
        contentBox.appendChild(detailsEl);
        overlay.appendChild(contentBox);
        container.appendChild(overlay);
    } else {
        messageEl = overlay.querySelector('.feedback-message');
        detailsEl = overlay.querySelector('.feedback-details');
        
        // Update border color for existing overlay
        const contentBox = overlay.querySelector('div');
        if (contentBox) {
            contentBox.style.border = `2px solid ${borderColor}`;
        }
    }
    
    // Update message content and color
    if (messageEl) {
        messageEl.textContent = `${icon} ${message}`;
        messageEl.style.color = textColor;
    }
    if (detailsEl) {
        detailsEl.textContent = details || '';
    }
    
    // Show overlay
    overlay.style.display = 'flex';
    
    // Auto-hide after duration
    setTimeout(() => {
        overlay.style.display = 'none';
    }, duration);
}

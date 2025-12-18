/**
 * Base API utilities for making REST calls to fbTools backend.
 * Provides common error handling, request formatting, and response parsing.
 */

export class APIError extends Error {
    constructor(message, statusCode, response) {
        super(message);
        this.name = "APIError";
        this.statusCode = statusCode;
        this.response = response;
    }
}

export class BaseAPI {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    /**
     * Make a POST request to the API
     * @param {string} endpoint - API endpoint path
     * @param {object} data - Request body data
     * @returns {Promise<object>} Response data
     */
    async post(endpoint, data) {
        try {
            const url = `${this.baseUrl}${endpoint}`;
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new APIError(
                    `POST ${endpoint} failed: ${response.statusText}`,
                    response.status,
                    errorText
                );
            }

            return await response.json();
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError(`Network error: ${error.message}`, 0, null);
        }
    }

    /**
     * Make a GET request to the API
     * @param {string} endpoint - API endpoint path
     * @param {object} params - URL query parameters
     * @returns {Promise<object>} Response data
     */
    async get(endpoint, params = {}) {
        try {
            const url = new URL(`${this.baseUrl}${endpoint}`, window.location.origin);
            Object.entries(params).forEach(([key, value]) => {
                if (value !== null && value !== undefined) {
                    url.searchParams.append(key, value);
                }
            });

            const response = await fetch(url);

            if (!response.ok) {
                const errorText = await response.text();
                throw new APIError(
                    `GET ${endpoint} failed: ${response.statusText}`,
                    response.status,
                    errorText
                );
            }

            return await response.json();
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError(`Network error: ${error.message}`, 0, null);
        }
    }

    /**
     * Handle API errors with user feedback
     * @param {Error} error - The error to handle
     * @param {string} operation - Operation name for error message
     * @param {object} app - ComfyUI app instance (optional)
     */
    handleError(error, operation = "Operation", app = null) {
        console.error(`${this.constructor.name} ${operation} error:`, error);
        
        if (app?.extensionManager?.toast) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: `${operation} Failed`,
                detail: error.message,
                life: 4000,
            });
        }
    }

    /**
     * Show success toast notification
     * @param {string} summary - Toast summary
     * @param {string} detail - Toast detail message
     * @param {object} app - ComfyUI app instance (optional)
     */
    showSuccess(summary, detail, app = null) {
        if (app?.extensionManager?.toast) {
            app.extensionManager.toast.add({
                severity: "success",
                summary,
                detail,
                life: 2000,
            });
        }
    }
}

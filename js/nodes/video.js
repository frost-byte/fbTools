/**
 * Frontend code for StoryVideoBatch node
 * Handles dynamic job_id dropdown updates when story_name changes
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Register extension for StoryVideoBatch node
app.registerExtension({
    name: "fbTools.StoryVideoBatch",
    
    async nodeCreated(node) {
        // Only apply to StoryVideoBatch nodes
        if (node.comfyClass !== "fbt_StoryVideoBatch") {
            return;
        }
        
        console.log("fb_tools -> StoryVideoBatch: Node created, setting up dynamic job_id update");
        
        // Find the widgets
        const storyNameWidget = node.widgets?.find(w => w.name === 'story_name');
        const jobIdWidget = node.widgets?.find(w => w.name === 'job_id');
        
        if (!storyNameWidget || !jobIdWidget) {
            console.warn("fb_tools -> StoryVideoBatch: Required widgets not found", {
                storyNameWidget: !!storyNameWidget,
                jobIdWidget: !!jobIdWidget
            });
            return;
        }
        
        /**
         * Fetch job IDs for a given story from the API
         * @param {string} storyName - Name of the story
         * @returns {Promise<string[]>} Array of job ID strings
         */
        async function fetchJobIds(storyName) {
            if (!storyName) {
                console.log("fb_tools -> StoryVideoBatch: Empty story name, returning empty list");
                return [""];
            }
            
            try {
                console.log(`fb_tools -> StoryVideoBatch: Fetching job_ids for story='${storyName}'`);
                
                // Call the custom API endpoint
                const response = await api.fetchApi(`/fbtools/story/job_ids?story_name=${encodeURIComponent(storyName)}`);
                
                if (!response.ok) {
                    console.error(`fb_tools -> StoryVideoBatch: API error ${response.status}:`, await response.text());
                    return [""];
                }
                
                const data = await response.json();
                const jobIds = data.job_ids || [];
                
                console.log(`fb_tools -> StoryVideoBatch: Received ${jobIds.length} job_ids for story='${storyName}':`, jobIds);
                
                // Include empty string as first option (means "use most recent")
                return jobIds.length > 0 ? ["", ...jobIds] : [""];
            } catch (error) {
                console.error("fb_tools -> StoryVideoBatch: Error fetching job_ids:", error);
                return [""];
            }
        }
        
        /**
         * Update the job_id combo widget with new options
         * @param {string[]} jobIds - Array of job ID strings
         */
        function updateJobIdOptions(jobIds) {
            const currentValue = jobIdWidget.value;
            
            // Update the widget options
            jobIdWidget.options.values = jobIds;
            
            console.log(`fb_tools -> StoryVideoBatch: Updated job_id options to ${jobIds.length} items:`, jobIds);
            
            // Preserve current selection if it's still valid, otherwise select first (empty = most recent)
            if (jobIds.includes(currentValue)) {
                jobIdWidget.value = currentValue;
                console.log(`fb_tools -> StoryVideoBatch: Kept existing job_id='${currentValue}'`);
            } else {
                jobIdWidget.value = jobIds[0] || "";
                console.log(`fb_tools -> StoryVideoBatch: Reset job_id to '${jobIdWidget.value}' (previous value '${currentValue}' not found)`);
            }
        }
        
        // Store original callback if it exists
        const originalCallback = storyNameWidget.callback;
        
        // Override the story_name widget callback to update job_id options
        storyNameWidget.callback = async function(value) {
            console.log(`fb_tools -> StoryVideoBatch: story_name changed to '${value}'`);
            
            // Call original callback if it exists
            if (originalCallback) {
                originalCallback.apply(this, arguments);
            }
            
            // Fetch and update job_id options for the new story
            const jobIds = await fetchJobIds(value);
            updateJobIdOptions(jobIds);
        };
        
        // Initialize: Load job_ids for the currently selected story
        const initialStory = storyNameWidget.value;
        if (initialStory) {
            console.log(`fb_tools -> StoryVideoBatch: Initializing with story='${initialStory}'`);
            const jobIds = await fetchJobIds(initialStory);
            updateJobIdOptions(jobIds);
        }
    }
});

/**
 * Collapsible Info Sections for AI Dashboard
 * ==========================================
 * 
 * JavaScript functionality to handle collapsible information sections
 * that can be toggled by clicking on module titles.
 */

// Initialize collapsible functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCollapsibleSections();
});

function initializeCollapsibleSections() {
    // Find all clickable titles
    const clickableTitles = document.querySelectorAll('[id^="title-"]');
    
    clickableTitles.forEach(title => {
        title.addEventListener('click', function() {
            const infoId = this.id.replace('title-', '');
            const collapseElement = document.getElementById(`collapse-${infoId}`);
            
            if (collapseElement) {
                toggleCollapse(collapseElement, this);
            }
        });
        
        // Add hover effect
        title.addEventListener('mouseenter', function() {
            this.style.opacity = '0.8';
        });
        
        title.addEventListener('mouseleave', function() {
            this.style.opacity = '1';
        });
    });
}

function toggleCollapse(element, titleElement) {
    const isVisible = element.style.display !== 'none';
    
    if (isVisible) {
        // Hide the element
        element.style.display = 'none';
        titleElement.style.opacity = '1';
    } else {
        // Show the element with animation
        element.style.display = 'block';
        element.style.opacity = '0';
        element.style.transform = 'translateY(-10px)';
        
        // Animate in
        setTimeout(() => {
            element.style.transition = 'all 0.3s ease-in-out';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 10);
        
        titleElement.style.opacity = '0.9';
    }
}

// Re-initialize when Dash updates the layout
if (window.dash_clientside) {
    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.namespace = window.dash_clientside.namespace || {};
    
    window.dash_clientside.namespace.reinitialize_collapsible = function() {
        setTimeout(initializeCollapsibleSections, 100);
        return '';
    };
}

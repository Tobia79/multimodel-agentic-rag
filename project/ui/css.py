custom_css = """
    /* ============================================
       MAIN CONTAINER
       ============================================ */
    .progress-text { 
        display: none !important;
    }
    
    .gradio-container { 
        max-width: 1000px !important;
        width: 100% !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background: #f5f5f5 !important;
        color: #111111 !important;
    }
    
    /* ============================================
       TABS
       ============================================ */
    button[role="tab"] {
        color: #6b7280 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        transition: all 0.2s ease !important;
        background: transparent !important;
    }
    
    button[role="tab"]:hover {
        color: #111111 !important;
    }
    
    button[role="tab"][aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 2px solid #000000 !important;
        border-radius: 0 !important;
        background: transparent !important;
    }
    
    .tabs {
        border-bottom: none !important;
        border-radius: 0 !important;
        background: #f5f5f5 !important;
    }
    
    .tab-nav {
        border-bottom: 1px solid #e0e0e0 !important;
        border-radius: 0 !important;
        background: #f5f5f5 !important;
    }
    
    button[role="tab"]::before,
    button[role="tab"]::after,
    .tabs::before,
    .tabs::after,
    .tab-nav::before,
    .tab-nav::after {
        display: none !important;
        content: none !important;
        border-radius: 0 !important;
    }
    
    #doc-management-tab {
        max-width: 500px !important;
        margin: 0 auto !important;
        background: #f5f5f5 !important;
    }
    
    /* ============================================
       BUTTONS
       ============================================ */
    button {
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    
    .primary {
        background: #3b82f6 !important;
        color: white !important;
    }
    
    .primary:hover {
        background: #2563eb !important;
        transform: translateY(-1px) !important;
    }
    
    .stop {
        background: #ef4444 !important;
        color: white !important;
    }
    
    .stop:hover {
        background: #dc2626 !important;
        transform: translateY(-1px) !important;
    }
    
    /* ============================================
       CHAT INPUT BOX
       ============================================ */
    textarea[placeholder="输入消息…"],
    textarea[data-testid*="textbox"]:not(#file-list-box textarea) {
        background: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        color: #111111 !important;
    }
    
    textarea[placeholder="输入消息…"]:focus {
        background: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .gr-text-input:has(textarea[placeholder="输入消息…"]),
    [class*="chatbot"] + * [data-testid="textbox"],
    form:has(textarea[placeholder="输入消息…"]) > div {
        background: #ffffff !important;
        border: none !important;
        gap: 12px !important;
    }
    
    form:has(textarea[placeholder="输入消息…"]) button,
    [class*="chatbot"] ~ * button[type="submit"] {
        background: transparent !important;
        border: none !important;
        padding: 8px !important;
    }
    
    form:has(textarea[placeholder="输入消息…"]) button:hover {
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    form:has(textarea[placeholder="输入消息…"]) {
        gap: 12px !important;
        display: flex !important;
        background: #ffffff !important;
    }
    
    /* ============================================
       FILE UPLOAD
       ============================================ */
    .file-preview, 
    [data-testid="file-upload"] {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 5px !important;
        color: #111111 !important;
        min-height: 200px !important;
    }
    
    .file-preview:hover, 
    [data-testid="file-upload"]:hover {
        border-color: #3b82f6 !important;
        background: #fafafa !important;
    }
    
    .file-preview *,
    [data-testid="file-upload"] * {
        color: #111111 !important;
    }
    
    .file-preview .label,
    [data-testid="file-upload"] .label {
        display: none !important;
    }
    
    /* ============================================
       INPUTS & TEXTAREAS
       ============================================ */
    input, 
    textarea {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        color: #111111 !important;
        transition: border-color 0.2s ease !important;
    }
    
    input:focus, 
    textarea:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    textarea[readonly] {
        background: #fafafa !important;
        color: #374151 !important;
    }
    
    /* ============================================
       FILE LIST BOX
       ============================================ */
    #file-list-box {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    
    #file-list-box textarea {
        background: transparent !important;
        border: none !important;
        color: #111111 !important;
        padding: 0 !important;
    }
    
    /* ============================================
       CHATBOT CONTAINER
       ============================================ */
    .chatbot {
        border-radius: 5px !important;
        background: #ffffff !important;
        border: 1px solid #e8e8e8 !important;
    }

    .chatbot .message-wrap,
    .chatbot > div {
        gap: 8px !important;
        padding: 12px !important;
        background: #ffffff !important;
    }

    /* ============================================
       MESSAGE BUBBLES
       ============================================ */
    .message {
        border-radius: 10px !important;
    }

    .message.user {
        background: #3b82f6 !important;
        color: #ffffff !important;
    }
    
    .message.bot {
        background: #f3f4f6 !important;
        color: #111111 !important;
        border: 1px solid #e5e7eb !important;
        width: fit-content !important;
        max-width: 90% !important;
    }
    
    .message-row img {
        margin: 0px !important;
    }

    .avatar-container img {
        padding: 0px !important;
    }

    /* ============================================
       PROGRESS BAR
       ============================================ */
    .progress-bar-wrap {
        border-radius: 10px !important;
        overflow: hidden !important;
        background: #e5e7eb !important;
    }

    .progress-bar {
        border-radius: 10px !important;
        background: #3b82f6 !important;
    }
    
    /* ============================================
       TYPOGRAPHY
       ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    p, label, span, .markdown, .prose {
        color: #111111 !important;
    }
    
    /* ============================================
       GLOBAL OVERRIDES
       ============================================ */
    * {
        box-shadow: none !important;
    }
    
    footer {
        visibility: hidden;
    }
"""

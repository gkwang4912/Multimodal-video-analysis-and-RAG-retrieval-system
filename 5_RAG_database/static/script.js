document.addEventListener('DOMContentLoaded', () => {
    // Top level elements
    const videoInput = document.getElementById('videoInput');
    const mainVideo = document.getElementById('mainVideo');
    const videoPlaceholder = document.getElementById('videoPlaceholder');

    // Chat elements
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatContainer = document.getElementById('chatContainer');

    // 1. Video Handling
    const videoContainer = document.querySelector('.group.relative'); // Select wrapper
    const fileInput = document.getElementById('videoInput');

    function handleFile(file) {
        if (file && file.type.startsWith('video/')) {
            const objectUrl = URL.createObjectURL(file);
            mainVideo.src = objectUrl;
            videoPlaceholder.style.opacity = '0';
            setTimeout(() => {
                videoPlaceholder.classList.add('hidden');
                mainVideo.play();
            }, 300);
        } else {
            alert('請上傳有效的影片檔案');
        }
    }

    videoInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

    // Drag and Drop
    const dropZone = document.getElementById('videoPlaceholder'); // Use the placeholder as the drop zone

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('bg-gray-800/90', 'border-blue-500', 'border-2', 'border-dashed');
    }

    function unhighlight(e) {
        dropZone.classList.remove('bg-gray-800/90', 'border-blue-500', 'border-2', 'border-dashed');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFile(files[0]);
    }

    // 2. Chat Logic
    function addMessage(text, isUser = false, sources = null) {
        const div = document.createElement('div');
        div.className = `flex gap-4 message-anim ${isUser ? 'flex-row-reverse' : ''}`;

        let contentHtml = '';
        let avatarHtml = '';

        if (isUser) {
            avatarHtml = `
                <div class="w-8 h-8 rounded-lg bg-gray-700 flex-shrink-0 flex items-center justify-center mt-1">
                    <i class="fa-solid fa-user text-gray-300 text-xs"></i>
                </div>
            `;
            contentHtml = `
                <div class="space-y-1 max-w-[85%] flex flex-col items-end">
                    <div class="bg-blue-600/90 text-white p-4 rounded-2xl rounded-tr-none shadow-sm">
                        <p class="message-content text-sm">${text}</p>
                    </div>
                </div>
            `;
        } else {
            avatarHtml = `
                <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex-shrink-0 flex items-center justify-center mt-1">
                    <i class="fa-solid fa-robot text-white text-xs"></i>
                </div>
            `;

            // Prepare Sources HTML
            let sourcesHtml = '';
            if (sources && (sources.images.length > 0 || sources.text.length > 0)) {

                // Images
                let imagesHtml = '';
                if (sources.images.length > 0) {
                    imagesHtml = `<div class="flex gap-2 mb-3 overflow-x-auto pb-2 scrollbar-thin">`;
                    sources.images.forEach(img => {
                        imagesHtml += `
                            <div class="relative w-24 h-24 flex-shrink-0 rounded-lg overflow-hidden border border-white/10 cursor-pointer group" onclick="showImageModal('${img.url}')">
                                <img src="${img.url}" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500">
                                <div class="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors"></div>
                            </div>
                        `;
                    });
                    imagesHtml += `</div>`;
                }

                // Texts (References)
                let textRefsHtml = '';
                if (sources.text.length > 0) {
                    textRefsHtml = `<div class="mt-3 text-xs text-gray-500 border-t border-gray-700/50 pt-2">
                        <div class="font-medium mb-1">參考來源:</div>
                        <ul class="space-y-1">`;
                    // Limit to top 3 for UI cleanliness
                    sources.text.slice(0, 3).forEach((txt, i) => {
                        textRefsHtml += `<li class="truncate hover:text-gray-300 transition-colors" title="${txt.content}">[${i + 1}] ${txt.chunk_id}</li>`;
                    });
                    textRefsHtml += `</ul></div>`;
                }

                sourcesHtml = `
                    <div class="mt-4 pt-3 border-t border-white/5">
                        ${imagesHtml}
                        ${textRefsHtml}
                    </div>
                `;
            }

            contentHtml = `
                <div class="space-y-1 max-w-[85%]">
                    <div class="text-xs text-gray-400 font-medium ml-1">AI Assistant</div>
                    <div class="bg-dark-surface border border-dark-border p-4 rounded-2xl rounded-tl-none text-gray-200 shadow-sm">
                        <div class="prose text-sm text-gray-300 leading-relaxed">${formatMarkdown(text)}</div>
                        ${sourcesHtml}
                    </div>
                </div>
            `;
        }

        div.innerHTML = isUser ? contentHtml + avatarHtml : avatarHtml + contentHtml;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addLoadingIndicator() {
        const div = document.createElement('div');
        div.id = 'loadingMsg';
        div.className = 'flex gap-4 message-anim';
        div.innerHTML = `
            <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex-shrink-0 flex items-center justify-center mt-1">
                <i class="fa-solid fa-robot text-white text-xs"></i>
            </div>
            <div class="space-y-1">
                <div class="text-xs text-gray-400 font-medium ml-1">AI Assistant</div>
                <div class="bg-dark-surface border border-dark-border px-4 py-3 rounded-2xl rounded-tl-none flex items-center gap-1.5">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function removeLoadingIndicator() {
        const el = document.getElementById('loadingMsg');
        if (el) el.remove();
    }

    // Simple markdown formatter replacement
    function formatMarkdown(text) {
        if (!text) return '';
        // Bold
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Lists
        text = text.replace(/^\s*-\s+(.*)$/gm, '<ul><li>$1</li></ul>');
        // Paragraphs (double newline)
        text = text.replace(/\n\n/g, '<br><br>');
        return text;
    }

    // Modal
    window.showImageModal = function (src) {
        const modal = document.getElementById('imageModal');
        const img = document.getElementById('modalImage');
        img.src = src;
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }

    // Send Message
    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text) return;

        // Reset input
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Add User Message
        addMessage(text, true);

        // API Call
        addLoadingIndicator();
        sendBtn.disabled = true;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await response.json();

            removeLoadingIndicator();

            if (data.error) {
                addMessage(`Error: ${data.error}`, false);
            } else {
                addMessage(data.answer, false, data.sources);
            }

        } catch (error) {
            removeLoadingIndicator();
            addMessage("抱歉，連線發生錯誤，請檢查後端伺服器是否啟動。", false);
            console.error(error);
        } finally {
            sendBtn.disabled = false;
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

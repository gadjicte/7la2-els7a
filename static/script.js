const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const msgsArea = document.getElementById('msgs');

// Helper: Get Time String
function getTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Helper: Append Message
function appendMessage(name, img, side, text) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', side);

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('msg-content');
    contentDiv.innerText = text;

    const timeDiv = document.createElement('div');
    timeDiv.classList.add('msg-time');
    timeDiv.innerText = getTime();

    msgDiv.appendChild(contentDiv);
    msgDiv.appendChild(timeDiv);

    msgsArea.appendChild(msgDiv);
    msgsArea.scrollTop = msgsArea.scrollHeight;
}

// Handle Submit
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = userInput.value;
    if (!text) return;

    // 1. Show User Message
    appendMessage('User', '', 'user', text);
    userInput.value = '';

    // 2. Show Typing Indicator (Temporary)
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot');
    typingDiv.innerHTML = `
        <div class="msg-content" style="color: #888; font-style: italic;">
            Thinking...
        </div>
    `;
    msgsArea.appendChild(typingDiv);
    msgsArea.scrollTop = msgsArea.scrollHeight;

    try {
        // 3. Send to Backend
        const response = await fetch('/get', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `msg=${encodeURIComponent(text)}`
        });

        const data = await response.json();
        
        // Remove Typing Indicator
        msgsArea.removeChild(typingDiv);

        // 4. Show Bot Response
        appendMessage('Bot', '', 'bot', data.response);

    } catch (error) {
        msgsArea.removeChild(typingDiv);
        appendMessage('Bot', '', 'bot', "Sorry, I couldn't reach the server.");
        console.error(error);
    }
});

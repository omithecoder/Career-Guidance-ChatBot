document.addEventListener('DOMContentLoaded', function () {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const modeSelectionButtons = document.querySelectorAll('.mode-selection button');

    let selectedMode = '';

    modeSelectionButtons.forEach((button) => {
        button.addEventListener('click', function () {
            selectedMode = button.textContent;
            chatMessages.innerHTML = `Selected mode: ${selectedMode}`;
        });
    });

    sendButton.addEventListener('click', function () {
        const message = userInput.value;
        if (message) {
            chatMessages.innerHTML += `<p>User: ${message}</p>`;
            if (selectedMode) {
                // Implement chatbot logic based on the selected mode
                chatMessages.innerHTML += `<p>Chatbot: [Response based on ${selectedMode}]</p>`;
            }
            userInput.value = '';
        }
    });
});

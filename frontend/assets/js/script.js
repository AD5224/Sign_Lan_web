// Automatically fetch translations and update UI every 3 seconds
setInterval(() => {
    fetch('/predict')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                document.getElementById('english-text').innerText = data.english;
                document.getElementById('hindi-text').innerText = data.hindi;
                document.getElementById('marathi-text').innerText = data.marathi;
            }
        })
        .catch(error => console.error('Error fetching prediction:', error));
}, 3000);

// Play English speech
function playEnglish() {
    const audio = new Audio('/speak/english');
    audio.play();
}

// Play Hindi speech
function playHindi() {
    const audio = new Audio('/speak/hindi');
    audio.play();
}

// Play Marathi speech
function playMarathi() {
    const audio = new Audio('/speak/marathi');
    audio.play();
}

// Add Space to the sentence
function addSpace() {
    fetch('/add_space', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log('Space added!');
        })
        .catch(error => console.error('Error adding space:', error));
}

// Clear the sentence
function clearSentence() {
    fetch('/clear_sentence', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log('Sentence cleared!');
            // Also clear output immediately
            document.getElementById('english-text').innerText = '';
            document.getElementById('hindi-text').innerText = '';
            document.getElementById('marathi-text').innerText = '';
        })
        .catch(error => console.error('Error clearing sentence:', error));
}

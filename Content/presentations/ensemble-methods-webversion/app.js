// Tab navigation
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');
const progressFill = document.getElementById('progressFill');

const tabOrder = ['intro', 'bagging', 'boosting', 'voting', 'stacking', 'advanced', 'lab', 'comparison', 'quiz'];

function switchTab(tabName) {
  // Remove active class from all tabs and contents
  tabs.forEach(tab => tab.classList.remove('active'));
  tabContents.forEach(content => content.classList.remove('active'));
  
  // Add active class to selected tab and content
  const selectedTab = document.querySelector(`[data-tab="${tabName}"]`);
  const selectedContent = document.getElementById(tabName);
  
  if (selectedTab && selectedContent) {
    selectedTab.classList.add('active');
    selectedContent.classList.add('active');
    
    // Update progress bar
    const tabIndex = tabOrder.indexOf(tabName);
    const progress = ((tabIndex + 1) / tabOrder.length) * 100;
    progressFill.style.width = `${progress}%`;
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const tabName = tab.getAttribute('data-tab');
    switchTab(tabName);
  });
});

// Copy code functionality
const copyButtons = document.querySelectorAll('.copy-btn');

copyButtons.forEach(button => {
  button.addEventListener('click', () => {
    const codeType = button.getAttribute('data-code');
    const codeElement = document.getElementById(`code-${codeType}`);
    
    if (codeElement) {
      const code = codeElement.textContent;
      
      // Create temporary textarea to copy text
      const textarea = document.createElement('textarea');
      textarea.value = code;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      
      try {
        document.execCommand('copy');
        button.textContent = 'Copied!';
        button.classList.add('copied');
        
        setTimeout(() => {
          button.textContent = 'Copy Code';
          button.classList.remove('copied');
        }, 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
      }
      
      document.body.removeChild(textarea);
    }
  });
});

// Quiz functionality
const quizAnswers = {
  q1: 'b',  // Bagging trains models in parallel, boosting trains them sequentially
  q2: 'c',  // Stacking
  q3: 'b',  // Prediction probabilities
  q4: 'c',  // Adjusts weights on misclassified samples (that's boosting, not Random Forest)
  q5: 'b',  // To learn how to best combine base model predictions
  q6: 'c'   // CatBoost
};

const quizExplanations = {
  q1: 'Bagging (Bootstrap Aggregating) trains multiple models in parallel on different bootstrap samples, while boosting trains models sequentially, with each model focusing on the errors of previous models.',
  q2: 'According to our experimental results, Stacking achieved the highest accuracy at 83.33%, outperforming all other ensemble methods.',
  q3: 'Soft voting uses prediction probabilities from each model and averages them, while hard voting only uses the predicted class labels and takes the majority vote.',
  q4: 'Adjusting weights on misclassified samples is a characteristic of boosting methods like AdaBoost, not Random Forest. Random Forest uses bootstrap sampling and parallel training.',
  q5: 'The meta-model (or final estimator) in stacking learns to optimally combine the predictions from the base models, effectively learning which models to trust for different types of inputs.',
  q6: 'CatBoost (Categorical Boosting) is specifically designed to handle categorical features natively without requiring extensive preprocessing like one-hot encoding.'
};

const submitQuizButton = document.getElementById('submitQuiz');
const quizResults = document.getElementById('quizResults');
const scoreElement = document.getElementById('score');
const answersElement = document.getElementById('answers');

if (submitQuizButton) {
  submitQuizButton.addEventListener('click', () => {
    let correctCount = 0;
    const totalQuestions = Object.keys(quizAnswers).length;
    let resultsHTML = '';
    
    // Check each question
    Object.keys(quizAnswers).forEach((questionId, index) => {
      const selectedOption = document.querySelector(`input[name="${questionId}"]:checked`);
      const correctAnswer = quizAnswers[questionId];
      const explanation = quizExplanations[questionId];
      
      if (selectedOption) {
        const isCorrect = selectedOption.value === correctAnswer;
        if (isCorrect) correctCount++;
        
        resultsHTML += `
          <div class="answer-item ${isCorrect ? 'correct' : 'incorrect'}">
            <strong>Question ${index + 1}:</strong> ${isCorrect ? '✓ Correct' : '✗ Incorrect'}<br>
            ${!isCorrect ? `<em>Correct answer: ${correctAnswer.toUpperCase()}</em><br>` : ''}
            <small>${explanation}</small>
          </div>
        `;
      } else {
        resultsHTML += `
          <div class="answer-item incorrect">
            <strong>Question ${index + 1}:</strong> Not answered<br>
            <em>Correct answer: ${correctAnswer.toUpperCase()}</em><br>
            <small>${explanation}</small>
          </div>
        `;
      }
    });
    
    // Calculate percentage
    const percentage = (correctCount / totalQuestions) * 100;
    
    // Display results
    scoreElement.textContent = `You scored ${correctCount} out of ${totalQuestions} (${percentage.toFixed(1)}%)`;
    
    if (percentage === 100) {
      scoreElement.textContent += ' - Perfect score! 🎉';
    } else if (percentage >= 80) {
      scoreElement.textContent += ' - Great job! 👏';
    } else if (percentage >= 60) {
      scoreElement.textContent += ' - Good effort! 👍';
    } else {
      scoreElement.textContent += ' - Review the material and try again! 📚';
    }
    
    answersElement.innerHTML = resultsHTML;
    quizResults.style.display = 'block';
    
    // Scroll to results
    quizResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
}

// Table sorting functionality
const sortableHeaders = document.querySelectorAll('.sortable');
let currentSort = { column: null, ascending: true };

sortableHeaders.forEach(header => {
  header.addEventListener('click', () => {
    const sortType = header.getAttribute('data-sort');
    const table = header.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // Determine sort direction
    if (currentSort.column === sortType) {
      currentSort.ascending = !currentSort.ascending;
    } else {
      currentSort.column = sortType;
      currentSort.ascending = true;
    }
    
    // Get column index
    const headers = Array.from(table.querySelectorAll('th'));
    const columnIndex = headers.indexOf(header);
    
    // Sort rows
    rows.sort((a, b) => {
      const aValue = a.cells[columnIndex].textContent.trim();
      const bValue = b.cells[columnIndex].textContent.trim();
      
      // Handle percentage values
      const aNum = parseFloat(aValue.replace('%', ''));
      const bNum = parseFloat(bValue.replace('%', ''));
      
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return currentSort.ascending ? aNum - bNum : bNum - aNum;
      }
      
      // String comparison
      if (currentSort.ascending) {
        return aValue.localeCompare(bValue);
      } else {
        return bValue.localeCompare(aValue);
      }
    });
    
    // Clear and repopulate tbody
    tbody.innerHTML = '';
    rows.forEach(row => tbody.appendChild(row));
  });
});

// Initialize progress bar
window.addEventListener('load', () => {
  progressFill.style.width = '11.11%'; // First tab (intro)
});
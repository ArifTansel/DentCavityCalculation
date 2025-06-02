document.addEventListener('DOMContentLoaded', function() {
    // Fetch STL file options from the server
    fetch('http://localhost:3000/api/student-list')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(stlOptions => {
        // Get reference to the select element
        const selectElement = document.getElementById('optionSelect');
        
        // Clear any existing options except the first one
        while (selectElement.options.length > 1) {
          selectElement.remove(1);
        }
        
        // Add the fetched options to the select element
        stlOptions.forEach(file => {
          const option = document.createElement('option');
          option.value = file.name;
          option.textContent = file.name;
          selectElement.appendChild(option);
        });
      })
      .catch(error => {
        console.error('Error fetching STL files:', error);
      });
  });

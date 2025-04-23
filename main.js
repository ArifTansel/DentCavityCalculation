const express = require('express');
const fs = require('fs');
const { exec } = require('child_process');
const path = require('path');
const app = express();
const port = 3000;

app.use(express.json());
// Directory path
const directoryPath = path.join(__dirname, 'StudentTeeth');
const outputFilesPath = path.join(__dirname, 'output'); 
const inputFilesPath = path.join(__dirname, 'input');
// Function to get STL file options
function getSTLFileOptions() {
  return new Promise((resolve, reject) => {
    fs.readdir(directoryPath, (err, files) => {
      if (err) {
        console.error('Error reading directory:', err);
        return reject(err);
      }
      
      // Filter for .stl files and create objects for the select options
      const stlFiles = files
        .filter(file => path.extname(file).toLowerCase() === '.stl')
        .map((file, index) => {
          // Remove the .stl extension to get a cleaner name
          const fileName = path.basename(file, '.stl');
          return {
            value: file,       // Full filename with extension (for the value)
            name: fileName     // Filename without extension (for display)
          };
        });
      
      resolve(stlFiles);
    });
  });
}

// API endpoint to get STL files
app.get('/api/stl-files', async (req, res) => {
  try {
    const stlOptions = await getSTLFileOptions();
    res.json(stlOptions);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve STL files' });
  }
});

app.get('/output/stl-files', async (req, res) => {
    try {
      const stlOptions = await getSTLFileOptions();
      res.json(stlOptions);
    } catch (error) {
      res.status(500).json({ error: 'Failed to retrieve STL files' });
    }
  });


app.post('/api/select', (req, res) => {
    const selectedFile = req.body.selectedOption;
    console.log(selectedFile)
    if (!selectedFile) {
        return res.status(400).json({ error: 'No option provided' });
    }

    const command = `python main.py --path ${selectedFile}`;


    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Execution error: ${error}`);
            return res.status(500).json({ error: 'Command execution failed', details: stderr });
        }
        console.log(`Command output: ${stdout}`);
        const outputJson = JSON.parse(stdout);
        res.json(outputJson);  //buraya bak
    });
});
// Serve files from the output directory
app.use('/output', express.static(outputFilesPath));
app.use('/input', express.static(inputFilesPath));

// Optionally serve the HTML page
app.use(express.static('public')); // Put your HTML file in a 'public' folder

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
const express = require("express");
const fs = require("fs");
const { exec } = require("child_process");
const path = require("path");
const mysql = require("mysql2");
const multer = require('multer');
const app = express();
const port = 3000;

app.use(express.json());
// database connection
const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "patatoes",
  database: "cavity_analysis_db",
});

db.connect((err) => {
  if (err) throw err;
  console.log("Veritabanına bağlanıldı.");
});

// Directory path
const outputFilesPath = path.join(__dirname, "output");
const inputFilesPath = path.join(__dirname, "input");
const publicDir = path.join(__dirname, 'public'); 


app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
function getStudentOptions() {
  return new Promise((resolve, reject) => {
    const query =
      "SELECT studentID, studentName, studentLastname FROM student_list";

    db.query(query, (err, results) => {
      if (err) {
        console.error("Error executing query:", err);
        return reject(err);
      }

      const options = results.map((row) => ({
        value: row.studentID, // Öğrenci numarası
        name: `${row.studentName} ${row.studentLastname} (${row.studentID})`, // Görüntülenecek isim
      }));

      resolve(options);
    });
  });
}

// API endpoint to get STL files
app.get("/api/student-list", async (req, res) => {
  try {
    const stlOptions = await getStudentOptions();
    res.json(stlOptions);
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Failed to retrieve STL files" });
  }
});

// POST endpoint
app.post("/api/user-file", (req, res) => {
  const studentId = String(req.body.selectedStudentID);

  if (!studentId) {
    return res
      .status(400)
      .json({ success: false, message: "Student ID eksik" });
  }
  const query = "SELECT s.stlFile, c.* FROM student_list s LEFT JOIN cavity_scores c ON s.studentID = c.studentID WHERE s.studentID = ?;";
  db.query(query, [studentId], (err, results) => {
    if (err) {
      console.error("Sorgu hatası:", err);
      return res.status(500).json({ success: false, message: "Sunucu hatası" });
    }
  return res.json({result : results  })

  });
  

});


app.get('/class', (req, res) => {
  const query = 'SELECT sl.studentID, sl.studentName, sl.studentLastname, sl.stlFile, cs.score FROM student_list sl LEFT JOIN cavity_scores cs ON sl.studentID = cs.studentID';

  db.query(query, (err, results) => {
    if (err) {
      console.error('DB Hatası:', err);
      return res.status(500).send('Sunucu hatası');
    }

    // results dizisi ile EJS dosyasını render et
    res.render('class', { students: results });
  });
});



app.post("/api/calculate", (req, res) => {
  const studentId = req.body.selectedOption;
  console.log(studentId);
  if (!studentId) {
    return res.status(400).json({ error: "No option provided" });
  }

  const command = `python main.py --studentId ${studentId}`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Execution error: ${error}`);
      return res
        .status(500)
        .json({ error: "Command execution failed", details: stderr });
    }
    console.log(`Command output: ${stdout}`);
    const outputJson = JSON.parse(stdout);
    res.json(outputJson); //buraya bak
  });
});

const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, path.join(__dirname, 'StudentTeeth'));  
  },
  filename: function(req, file, cb) {
    const studentId = req.body.studentId.trim(); 
    cb(null, `${studentId}.stl`);
  }
});

const upload = multer({ storage: storage });
app.post('/uploadFile', upload.single('stlFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('Dosya yüklenmedi!');
  }
  const studentId = req.body.studentId.trim();
  const updateQuery = `UPDATE student_list SET stlFile = true WHERE studentID = ?`;
  
  db.query(updateQuery, [studentId], (err, result) => {
    if (err) {
      console.error("Veritabanı güncelleme hatası:", err);
      return res.status(500).json({ success: false, message: "Güncelleme başarısız." });
    }
    res.redirect('/class'); //dosya yüklendi
  });
});


/////////////////////
app.post('/addStudent', (req, res) => {
  const { studentID, studentName, studentLastname } = req.body;

  const insertQuery = `
    INSERT INTO student_list (studentID, studentName, studentLastname, stlFile)
    VALUES (?, ?, ?, false)
  `;

  db.query(insertQuery, [studentID, studentName, studentLastname], (err, result) => {
    if (err) {
      console.error("Veritabanına ekleme hatası:", err);
      return res.status(500).send("Ekleme işlemi başarısız");
    }
    res.redirect('/class'); // Başarılıysa liste sayfasına yönlendir
  });
});



app.get('/search', (req, res) => {
  const q = req.query.q;
  if (!q) return res.json([]);

  const sql = "SELECT * FROM student_list WHERE studentName LIKE ?";
  db.query(sql, [`%${q}%`], (err, results) => {
    res.json(results);
  });
});

// Serve files from the output directory
app.use("/output", express.static(outputFilesPath));
app.use("/input", express.static(inputFilesPath));

// Optionally serve the HTML page
app.use(express.static("public")); // Put your HTML file in a 'public' folder

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

//TODO öğrenci için PDF OLUŞTURMA BUTONU 
//TODO SINIF için PDF OLUŞTURMA BUTONU 

const searchInput = document.getElementById('searchInput');
const studentTable = document.getElementById('student-table');
const tableRows = studentTable.querySelectorAll('tbody tr');

searchInput.addEventListener('input', async () => {
  const query = searchInput.value.trim().toLowerCase();

  if (query === '') {
    // Arama boşsa tüm satırları göster
    tableRows.forEach(row => row.style.display = '');
    return;
  }

  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) throw new Error("Sunucu hatası");

    const data = await response.json();

    // Gelen tüm öğrenci ID'lerini listeye al
    const matchedIDs = data.map(student => student.studentID.toString().toLowerCase());

    // Her tablo satırını kontrol et
    tableRows.forEach(row => {
      const studentIdCell = row.querySelector('#student-id');
      const studentId = studentIdCell.textContent.trim().toLowerCase();

      if (matchedIDs.includes(studentId)) {
        row.style.display = '';  // göster
      } else {
        row.style.display = 'none'; // gizle
      }
    });
  } catch (err) {
    console.error("Arama sırasında hata:", err);
  }
});

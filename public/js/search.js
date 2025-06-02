const searchInput = document.getElementById('searchInput');
const resultList = document.getElementById('resultList');

searchInput.addEventListener('input', async () => {
  const query = searchInput.value.trim();
  resultList.innerHTML = '';

  if (query === '') return;

  const res = await fetch(`http://localhost:3000/search?q=${encodeURIComponent(query)}`);
  const data = await res.json();

  data.forEach(user => {
    const li = document.createElement('button');
    li.className = 'list-group-item';
    li.textContent = user.studentName; // Eğer name dışında bilgiler varsa ekle
    resultList.appendChild(li);
  });
});
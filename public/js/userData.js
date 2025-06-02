
document.getElementById("optionSelect").addEventListener("change", function () {
  const selectedValue = this.value;
  const match = selectedValue.match(/\((\d+)\)/);
  const fileUpload = document.getElementById("fileUpload");

  // Seçim "Choose an option..." ise işlem yapma
  if (selectedValue === "Choose an option...") return;
  const statusValue = document.getElementById("status");

  fetch("/api/user-file", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ selectedStudentID: match[1] }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Sunucu hatası");
      }
      return response.json(); // veya .text() backend cevabına göre
    })
    .then((data) => {
      // eğer kullanıcının path kaydı varsa True yoksa False cevabı gelecek
      console.log("Başarılı:", data);
      if (data.result[0].stlFile) { //var ise görüntüyü kapat
        // TODO kullanıcının path i varsa 1-) display none (done)  |  2-) kullanıcının not bilgileri çekilmeli
        // id = fileUpload kısmını görünür yap
        fileUpload.style.display = "none";
        const scoreData = data.result[0]
        writeData(scoreData)
        changeColors(scoreData)
      } else {
        deleteData()
        clearColors()
        fileUpload.style.display = "block";
      }
    })
    .catch((error) => {
      console.error("Hata:", error);
    });
});

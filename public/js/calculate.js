function writeData(data) {
  const right_angle = document.getElementById("right_angle");
  const left_angle = document.getElementById("left_angle");
  const cavity_depth = document.getElementById("cavity_depth");
  const roughness = document.getElementById("roughness");
  const b_l_length_ratio = document.getElementById("b_l_length_ratio");
  const b_l_length = document.getElementById("b_l_length");
  const m_d_length_ratio = document.getElementById("m_d_length_ratio");
  const m_d_length = document.getElementById("m_d_length");
  const distal_ridge_distance = document.getElementById(
    "distal_ridge_distance"
  );
  const mesial_ridge_distance = document.getElementById(
    "mesial_ridge_distance"
  );
  const distal_isthmus_length = document.getElementById(
    "distal_isthmus_length"
  );
  const mesial_isthmus_length = document.getElementById(
    "mesial_isthmus_length"
  );
  const score = document.getElementById("score");

  right_angle.innerText = data.right_angle;
  left_angle.innerText = data.left_angle;
  cavity_depth.innerText = data.cavity_depth + " mm";
  roughness.innerText = data.roughness;
  b_l_length_ratio.innerText = data.b_l_length_ratio;
  b_l_length.innerText = data.b_l_length;
  m_d_length_ratio.innerText = data.m_d_length_ratio;
  m_d_length.innerText = data.m_d_length;
  distal_ridge_distance.innerText = data.distal_ridge_distance;
  mesial_ridge_distance.innerText = data.mesial_ridge_distance;
  distal_isthmus_length.innerText = data.distal_isthmus_width;
  mesial_isthmus_length.innerText = data.mesial_isthmus_width;
  score.innerText = data.score;
}
function deleteData() {
  const right_angle = document.getElementById("right_angle");
  const left_angle = document.getElementById("left_angle");
  const cavity_depth = document.getElementById("cavity_depth");
  const roughness = document.getElementById("roughness");
  const b_l_length_ratio = document.getElementById("b_l_length_ratio");
  const m_d_length_ratio = document.getElementById("m_d_length_ratio");
  const distal_ridge_distance = document.getElementById(
    "distal_ridge_distance"
  );
  const mesial_ridge_distance = document.getElementById(
    "mesial_ridge_distance"
  );
  const score = document.getElementById("score");
  const m_d_length = document.getElementById("m_d_length");
  const b_l_length = document.getElementById("b_l_length");

  m_d_length.innerText = "--";
  b_l_length.innerText = "--";
  right_angle.innerText = "--";
  left_angle.innerText = "--";
  cavity_depth.innerText = "--";
  roughness.innerText = "--";
  b_l_length_ratio.innerText = "--";
  m_d_length_ratio.innerText = "--";
  distal_ridge_distance.innerText = "--";
  mesial_ridge_distance.innerText = "--";
  score.innerText = "--";
}
function clearColors() {
  const mappings = [
    { dataKey: "is_right_angle_true", elementId: "right_angle" },
    { dataKey: "is_left_angle_true", elementId: "left_angle" },
    { dataKey: "is_cavity_depth_true", elementId: "cavity_depth" },
    { dataKey: "is_roughness_true", elementId: "roughness" },
    { dataKey: "is_m_d_length_ratio_true", elementId: "m_d_length_ratio" },
    { dataKey: "is_m_d_length_true", elementId: "m_d_length" },
    { dataKey: "is_b_l_length_ratio_true", elementId: "b_l_length_ratio" },
    { dataKey: "is_b_l_length_true", elementId: "b_l_length" },
    {
      dataKey: "is_distal_ridge_distance_true",
      elementId: "distal_ridge_distance",
    },
    {
      dataKey: "is_mesial_ridge_distance_true",
      elementId: "mesial_ridge_distance",
    },

    { dataKey: 'is_mesial_isthmus_width_true', elementId: 'mesial_isthmus_length' },
    { dataKey: 'is_distal_isthmus_width_true', elementId: 'distal_isthmus_length' },
  ];
  mappings.forEach((item) => {
    const element = document.getElementById(item.elementId).parentElement;
    if (element) {
      element.classList = [];
    }
  });
}
function changeColors(data) {
  // Define the keys to check and their corresponding table row IDs
  const mappings = [
    { dataKey: "is_right_angle_true", elementId: "right_angle" },
    { dataKey: "is_left_angle_true", elementId: "left_angle" },
    { dataKey: "is_cavity_depth_true", elementId: "cavity_depth" },
    { dataKey: "is_roughness_true", elementId: "roughness" },
    { dataKey: "is_m_d_length_ratio_true", elementId: "m_d_length_ratio" },
    { dataKey: "is_m_d_length_true", elementId: "m_d_length" },
    { dataKey: "is_b_l_length_ratio_true", elementId: "b_l_length_ratio" },
    { dataKey: "is_b_l_length_true", elementId: "b_l_length" },
    {
      dataKey: "is_distal_ridge_distance_true",
      elementId: "distal_ridge_distance",
    },
    {
      dataKey: "is_mesial_ridge_distance_true",
      elementId: "mesial_ridge_distance",
    },

    { dataKey: 'is_mesial_isthmus_width_true', elementId: 'mesial_isthmus_length' },
    { dataKey: 'is_distal_isthmus_width_true', elementId: 'distal_isthmus_length' },
  ];

  // Process each item in the mappings
  mappings.forEach((item) => {
    // Get the value from data
    const value = data[item.dataKey];

    // Get the table cell element
    const element = document.getElementById(item.elementId);

    if (element) {
      // Find the parent row of the cell
      const row = element.parentElement;

      // Remove any existing color classes
      row.classList.remove(
        "table-danger",
        "table-warning",
        "table-success",
        "text-white"
      );

      // Apply color based on value
      if (value === 0) {
        row.classList.add("table-danger", "text-white"); // Red for 0
      } else if (value === 0.5) {
        row.classList.add("table-warning"); // Yellow for 0.5
      } else if (value === 1) {
        row.classList.add("table-success", "text-white"); // Green for 1
      }
    }
  });
}
// Hesaplama Islemi
document
  .getElementById("calculateButton")
  .addEventListener("click", function () {
    const selectedValue = document.getElementById("optionSelect").value;
    const statusValue = document.getElementById("status");
    statusValue.innerText = "Hesaplanıyor...";
    const match = selectedValue.match(/\((\d+)\)/);

    // Example POST request
    fetch("http://localhost:3000/api/calculate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ selectedOption: match[1] }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Success:", data);
        writeData(data);
        changeColors(data);
        (statusValue.innerText = "Hesaplandı öğrenci notu :  "), data.score;
      })
      .catch((error) => {
        statusValue.innerText = "ERRORR :  ";

        console.error("Error:", error);
      });
  });

// public/js/viewer.js
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Set up the scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);
const container = document.getElementById("viewer");
const containerWidth = container.clientWidth * 0.95;
const containerHeight = container.clientHeight * 0.9;

// Set up the camera
const camera = new THREE.PerspectiveCamera(
  75,
  containerWidth / containerHeight,
  0.1,
  1000
);
camera.position.z = 5;

// Set up the renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(containerWidth, containerHeight);
container.appendChild(renderer.domElement);

// Add lighting
const ambientLight = new THREE.AmbientLight(0x404040);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(1, 1, 1);
scene.add(directionalLight);

const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight2.position.set(-1, -1, -1);
scene.add(directionalLight2);

// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.25;

// Current mesh holder
let currentMesh = null;

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// Handle window resize or container resize
window.addEventListener("resize", () => {
  const newWidth = container.clientWidth;
  const newHeight = container.clientHeight || 500;

  camera.aspect = newWidth / newHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(newWidth, newHeight);
});

// Function to load and display an STL file
window.loadSTL = function (filename) {
  const loader = new PLYLoader();
  if (currentMesh) {
    scene.remove(currentMesh);
  }
  console.log(filename)
  const studentName = document.getElementById("optionSelect").value;
  loader.load(`output/${studentName}/${filename}.ply`, (geometry) => {
    // Create material and mesh
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);

    // Center the model
    // geometry.computeBoundingBox();
    // const boundingBox = geometry.boundingBox;
    // const center = new THREE.Vector3();
    // boundingBox.getCenter(center);
    // mesh.position.set(-center.x, -center.y, -center.z);

    // Scale the model to fit in view
    // const maxDim = Math.max(
    //   boundingBox.max.x - boundingBox.min.x,
    //   boundingBox.max.y - boundingBox.min.y,
    //   boundingBox.max.z - boundingBox.min.z
    // );
    // const scaleFactor = 10 / maxDim;
    // mesh.scale.set(scaleFactor, scaleFactor, scaleFactor);

    // Add to scene and store reference
    scene.add(mesh);
    currentMesh = mesh;
  });
};

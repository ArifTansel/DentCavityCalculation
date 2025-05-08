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


// Track meshes and the parent group
let meshes = new Map();
let modelGroup = null;

// Load multiple meshes by filenames array
window.loadSTLs = function(filenames) {
  const loader = new PLYLoader();
  const studentName = document.getElementById("optionSelect").value;
  
  // First clear any existing meshes
  clearMeshes();
  
  // Create a new group for this set of meshes
  modelGroup = new THREE.Group();
  scene.add(modelGroup);
  
  // Load each mesh
  let loadedCount = 0;
  filenames.forEach(filename => {
    loader.load(`output/${studentName}/${filename}.ply`, (geometry) => {
      // Create material and mesh
      const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);

      // Add to the group and store reference in map
      modelGroup.add(mesh);
      meshes.set(filename, mesh);
      
      // Check if all meshes are loaded to center the group
      loadedCount++;
      if (loadedCount === filenames.length) {
        centerGroup();
      }
    });
  });
};

// Center the entire group of meshes while maintaining their relative positions
function centerGroup() {
  if (!modelGroup || modelGroup.children.length === 0) return;
  
  // Create a bounding box for the entire group
  const boundingBox = new THREE.Box3().setFromObject(modelGroup);
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);
  
  // Move the entire group as one unit
  modelGroup.position.set(-center.x, -center.y, -center.z);
}

// Clear all meshes and group from the scene
window.clearMeshes = function() {
  if (modelGroup) {
    scene.remove(modelGroup);
    modelGroup = null;
  }
  meshes.clear();
};

// For backward compatibility
window.loadSTL = function(filename) {
  window.loadSTLs([filename]);
};

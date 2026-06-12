// --- 3D Visualizer Setup ---

const container = document.getElementById('canvas-container');

// Scene, Camera, Renderer
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0b0c10, 0.05);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 8;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.0;

// Shaders

// GLSL 3D Simplex Noise function (Ashima Arts)
const simplexNoise3D = `
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}
`;

const vertexShader = `
uniform float uTime;
uniform float uAuth;   // Authentic Wave (Smoothness)
uniform float uClone;  // Clone Anomaly (Jagged spikes)
uniform float uVib;    // Micro-vibrations
uniform float uFlat;   // Amplitude Flattening

varying vec3 vNormal;
varying float vDisplacement;
varying vec3 vPosition;

${simplexNoise3D}

void main() {
    vNormal = normal;
    
    // 1. Authentic Smooth Waves (Low freq sine + noise)
    float smoothNoise = snoise(position * 1.5 + uTime * 0.5);
    float wave = sin(position.y * 3.0 + uTime * 2.0) * cos(position.x * 2.0 - uTime);
    float authDisplacement = (smoothNoise + wave) * 0.2 * uAuth;
    
    // 2. Clone Anomaly Spikes (High freq noise)
    float jaggedNoise = snoise(position * 8.0 - uTime * 3.0);
    // Make them spike outwards primarily
    float cloneDisplacement = abs(jaggedNoise) * 0.5 * uClone;
    
    // 3. Micro-Vibrations (Very fast, localized high freq)
    float vibNoise = snoise(position * 15.0 + uTime * 15.0);
    // Localize vibration mask based on y position (e.g. equatorial band)
    float mask = smoothstep(0.5, 1.0, 1.0 - abs(position.y));
    float vibDisplacement = vibNoise * 0.15 * uVib * mask;
    
    // Combine displacements
    float totalDisplacement = authDisplacement + cloneDisplacement + vibDisplacement;
    
    // 4. Amplitude Flattening (Compression effect reduces overall displacement scale)
    float flatteningFactor = 1.0 - (uFlat * 0.8); // max 80% flattened
    totalDisplacement *= flatteningFactor;
    
    vDisplacement = totalDisplacement;
    
    vec3 newPosition = position + normal * totalDisplacement;
    
    // Also flatten the y-axis physical scale if compression is high
    newPosition.y *= (1.0 - (uFlat * 0.3));

    vPosition = newPosition;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
}
`;

const fragmentShader = `
uniform float uTime;
uniform float uClone;

varying vec3 vNormal;
varying float vDisplacement;
varying vec3 vPosition;

void main() {
    // Cinematic Fresnel Edge Lighting
    vec3 viewDirection = normalize(cameraPosition - vPosition);
    float fresnel = dot(viewDirection, vNormal);
    fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
    fresnel = pow(fresnel, 3.0);
    
    // Colors
    vec3 baseColorAuth = vec3(0.0, 0.5, 0.8); // Cyber Blue
    vec3 baseColorClone = vec3(0.1, 0.8, 0.2); // Toxic Green
    
    // Mix color based on the Clone anomaly intensity
    float mixFactor = clamp(uClone / 2.0, 0.0, 1.0);
    vec3 color = mix(baseColorAuth, baseColorClone, mixFactor);
    
    // Glowing effect based on displacement intensity
    float glow = clamp(vDisplacement * 2.0, 0.0, 1.0);
    color += vec3(glow * 0.5, glow * 0.5, glow * 0.5); // Add white hot spots
    
    // Apply fresnel rim light
    vec3 rimColor = mix(vec3(0.2, 0.6, 1.0), vec3(0.5, 1.0, 0.2), mixFactor);
    color += rimColor * fresnel * 1.5;
    
    // Ambient darkness adjustment
    color *= 0.8 + fresnel * 0.2;

    gl_FragColor = vec4(color, 1.0);
}
`;

// Uniforms
const uniforms = {
    uTime: { value: 0.0 },
    uAuth: { value: 1.0 },
    uClone: { value: 0.0 },
    uVib: { value: 0.0 },
    uFlat: { value: 0.0 }
};

// Geometry (Icosahedron for uniform sphere with detail)
const geometry = new THREE.IcosahedronGeometry(2, 64); // High vertex count for displacement

const material = new THREE.ShaderMaterial({
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    uniforms: uniforms,
    wireframe: true // Makes it look technical/cyber-like
});

const sphere = new THREE.Mesh(geometry, material);
scene.add(sphere);

// Add inner core to hide wireframe see-through back faces
const innerMat = new THREE.MeshBasicMaterial({ color: 0x050608 });
const innerSphere = new THREE.Mesh(new THREE.IcosahedronGeometry(1.95, 32), innerMat);
scene.add(innerSphere);

// --- Render Loop ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    
    const elapsedTime = clock.getElapsedTime();
    uniforms.uTime.value = elapsedTime;
    
    controls.update();
    renderer.render(scene, camera);
}
animate();

// --- Window Resize ---
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- UI Binding ---
const uis = {
    auth: { range: document.getElementById('auth-param'), label: document.getElementById('val-auth') },
    clone: { range: document.getElementById('clone-param'), label: document.getElementById('val-clone') },
    vib: { range: document.getElementById('vib-param'), label: document.getElementById('val-vib') },
    flat: { range: document.getElementById('flat-param'), label: document.getElementById('val-flat') }
};

function updateUniforms() {
    uniforms.uAuth.value = parseFloat(uis.auth.range.value);
    uniforms.uClone.value = parseFloat(uis.clone.range.value);
    uniforms.uVib.value = parseFloat(uis.vib.range.value);
    uniforms.uFlat.value = parseFloat(uis.flat.range.value);
    
    uis.auth.label.innerText = uniforms.uAuth.value.toFixed(2);
    uis.clone.label.innerText = uniforms.uClone.value.toFixed(2);
    uis.vib.label.innerText = uniforms.uVib.value.toFixed(2);
    uis.flat.label.innerText = uniforms.uFlat.value.toFixed(2);
    
    // Toggle wireframe color logic? Actually wireframe looks great. We will just change material.wireframe based on deepfake level maybe? 
    // We'll keep wireframe:true, it looks highly technical.
}

// Attach listeners
Object.values(uis).forEach(ui => {
    ui.range.addEventListener('input', updateUniforms);
});

// Presets
document.getElementById('btn-authentic').addEventListener('click', () => {
    uis.auth.range.value = 1.2;
    uis.clone.range.value = 0.0;
    uis.vib.range.value = 0.0;
    uis.flat.range.value = 0.0;
    updateUniforms();
});

document.getElementById('btn-lowquality').addEventListener('click', () => {
    uis.auth.range.value = 0.2;
    uis.clone.range.value = 2.5;
    uis.vib.range.value = 0.0;
    uis.flat.range.value = 0.0;
    updateUniforms();
});

document.getElementById('btn-deepfake').addEventListener('click', () => {
    uis.auth.range.value = 0.8;
    uis.clone.range.value = 0.3;
    uis.vib.range.value = 1.5; // High micro-vibration resonance
    uis.flat.range.value = 0.0;
    updateUniforms();
});

document.getElementById('btn-compressed').addEventListener('click', () => {
    uis.auth.range.value = 0.5;
    uis.clone.range.value = 0.0;
    uis.vib.range.value = 0.0;
    uis.flat.range.value = 0.9; // Highly flattened amplitude
    updateUniforms();
});
